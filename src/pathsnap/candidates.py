"""Candidate generation: for every GPS fix, find nearby directed road arcs.

A uniform grid over the trip's road network stores each arc in every cell its
(radius-expanded) bounding box touches, so a lookup is O(local arcs) instead of
O(all arcs).  Each observation keeps the closest MAX_CANDIDATES arcs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (CANDIDATE_CELL_M, CANDIDATE_FALLBACK_RADII_M,
                     CANDIDATE_RADIUS_M, MAX_CANDIDATES)
from .geo import to_local, vec_bearing_rad
from .io import Trip


@dataclass
class CandidateSet:
    """Flat candidate table for one trip, one row per (observation, arc)."""
    obs_idx: np.ndarray        # (R,) observation index for each row
    arc: np.ndarray            # (R,) arc id
    dist: np.ndarray           # (R,) perpendicular distance, metres
    t: np.ndarray              # (R,) projection fraction along the arc
    foot_x: np.ndarray         # (R,)
    foot_y: np.ndarray
    rank: np.ndarray           # (R,) 0 = closest candidate for that observation
    n_cand: np.ndarray         # (R,) candidates available for that observation

    def __len__(self):
        return len(self.arc)


class TripNetwork:
    """Local-metric view of one trip's road network."""

    def __init__(self, trip: Trip):
        self.trip = trip
        self.frame = (float(np.mean(trip.nodes[:, 0])), float(np.mean(trip.nodes[:, 1])))
        self.node_x, self.node_y = to_local(trip.nodes[:, 0], trip.nodes[:, 1], self.frame)
        self.ax = self.node_x[trip.arcs[:, 0]]
        self.ay = self.node_y[trip.arcs[:, 0]]
        self.bx = self.node_x[trip.arcs[:, 1]]
        self.by = self.node_y[trip.arcs[:, 1]]
        self.dx = self.bx - self.ax
        self.dy = self.by - self.ay
        self.len = np.hypot(self.dx, self.dy)
        self.bearing = vec_bearing_rad(self.dx, self.dy)
        self._grid = None
        self._mid_tree = None
        self._node_tree = None
        self._mid_buffer = 0.0
        self._node_arcs = None
        self._out_arcs = None
        self._in_arcs = None

    # -- spatial index ------------------------------------------------------
    def _build_grid(self):
        radius = CANDIDATE_RADIUS_M
        cell = CANDIDATE_CELL_M
        xmin = np.minimum(self.ax, self.bx) - radius
        xmax = np.maximum(self.ax, self.bx) + radius
        ymin = np.minimum(self.ay, self.by) - radius
        ymax = np.maximum(self.ay, self.by) + radius
        self._grid_origin = (float(xmin.min()), float(ymin.min()))
        cx0 = np.floor((xmin - self._grid_origin[0]) / cell).astype(np.int64)
        cx1 = np.floor((xmax - self._grid_origin[0]) / cell).astype(np.int64)
        cy0 = np.floor((ymin - self._grid_origin[1]) / cell).astype(np.int64)
        cy1 = np.floor((ymax - self._grid_origin[1]) / cell).astype(np.int64)
        nx = int(cx1.max()) + 2
        cells = {}
        n_arcs = len(self.ax)
        # long arcs touch many cells; most touch <6
        for i in range(n_arcs):
            for cx in range(cx0[i], cx1[i] + 1):
                base = cx * nx
                for cy in range(cy0[i], cy1[i] + 1):
                    key = base + cy
                    lst = cells.get(key)
                    if lst is None:
                        cells[key] = [i]
                    else:
                        lst.append(i)
        self._grid = cells
        self._grid_nx = nx
        self._cell = cell

    def _query_grid(self, x, y):
        if self._grid is None:
            self._build_grid()
        cell = self._cell
        cx = int(np.floor((x - self._grid_origin[0]) / cell))
        cy = int(np.floor((y - self._grid_origin[1]) / cell))
        return self._grid.get(cx * self._grid_nx + cy)

    def _mid_kd(self):
        if self._mid_tree is None:
            from scipy.spatial import cKDTree
            self._mid_tree = cKDTree(np.column_stack([(self.ax + self.bx) / 2,
                                                       (self.ay + self.by) / 2]))
            self._node_tree = cKDTree(np.column_stack([self.node_x, self.node_y]))
            self._mid_buffer = float(np.percentile(self.len, 95)) * 0.5
        return self._mid_tree

    # -- adjacency ---------------------------------------------------------
    def out_arcs(self):
        """For each node, list of outgoing arc ids."""
        if self._out_arcs is None:
            out = [[] for _ in range(len(self.node_x))]
            for a in range(len(self.ax)):
                out[self.trip.arcs[a, 0]].append(a)
            self._out_arcs = out
            self._in_arcs = [[] for _ in range(len(self.node_x))]
            for a in range(len(self.ax)):
                self._in_arcs[self.trip.arcs[a, 1]].append(a)
        return self._out_arcs

    def in_arcs(self):
        self.out_arcs()
        return self._in_arcs

    def node_arcs(self):
        """For each node, every arc incident to it (either direction)."""
        if self._node_arcs is None:
            node_arcs = [[] for _ in range(len(self.node_x))]
            for a, (u, v) in enumerate(self.trip.arcs):
                node_arcs[int(u)].append(a)
                node_arcs[int(v)].append(a)
            self._node_arcs = node_arcs
        return self._node_arcs

    # -- candidate search ---------------------------------------------------
    def _dist_to_arcs(self, ids, x, y):
        ax = self.ax[ids]; ay = self.ay[ids]
        bx = self.bx[ids]; by = self.by[ids]
        dx = bx - ax; dy = by - ay
        seg2 = dx * dx + dy * dy
        safe = np.where(seg2 > 1e-12, seg2, 1.0)
        t = ((x - ax) * dx + (y - ay) * dy) / safe
        t = np.clip(t, 0.0, 1.0)
        fx = ax + t * dx; fy = ay + t * dy
        return np.hypot(x - fx, y - fy), t, fx, fy

    def candidates(self, points_xy: np.ndarray, radius: float = CANDIDATE_RADIUS_M,
                   max_cand: int = MAX_CANDIDATES) -> CandidateSet:
        rows_obs, rows_arc = [], []
        rows_dist, rows_t, rows_fx, rows_fy = [], [], [], []
        n = len(points_xy)
        radius = float(radius)
        fallbacks = list(CANDIDATE_FALLBACK_RADII_M)
        for i in range(n):
            x = float(points_xy[i, 0]); y = float(points_xy[i, 1])
            ids = self._query_grid(x, y)
            best = None
            if ids is not None and len(ids):
                ids = np.asarray(ids, dtype=np.int64)
                d, t, fx, fy = self._dist_to_arcs(ids, x, y)
                keep = d <= radius
                ids = ids[keep]; d = d[keep]; t = t[keep]; fx = fx[keep]; fy = fy[keep]
                if len(ids):
                    best = (ids, d, t, fx, fy)
            for fb in fallbacks:
                if best is not None:
                    break
                self._mid_kd()
                mid = self._mid_tree.query_ball_point([x, y], fb + self._mid_buffer)
                node_hits = self._node_tree.query_ball_point([x, y], fb)
                ids_set = set(int(i) for i in mid)
                node_arcs = self.node_arcs()
                for nd in node_hits:
                    ids_set.update(node_arcs[nd])
                if not ids_set:
                    continue
                ids = np.asarray(sorted(ids_set), dtype=np.int64)
                d, t, fx, fy = self._dist_to_arcs(ids, x, y)
                keep = d <= fb
                ids = ids[keep]; d = d[keep]; t = t[keep]; fx = fx[keep]; fy = fy[keep]
                if len(ids):
                    best = (ids, d, t, fx, fy)
            if best is None:
                # last resort: exact distance to every arc (rare, only far-off fixes)
                ids = np.arange(len(self.ax), dtype=np.int64)
                d, t, fx, fy = self._dist_to_arcs(ids, x, y)
                k = min(max_cand, len(ids))
                sel = np.argpartition(d, k - 1)[:k]
                best = (ids[sel], d[sel], t[sel], fx[sel], fy[sel])
            ids, d, t, fx, fy = best
            if len(ids) > max_cand:
                sel = np.argpartition(d, max_cand - 1)[:max_cand]
                ids, d, t, fx, fy = ids[sel], d[sel], t[sel], fx[sel], fy[sel]
            order = np.argsort(d, kind="stable")
            ids, d, t, fx, fy = ids[order], d[order], t[order], fx[order], fy[order]
            rows_obs.append(np.full(len(ids), i, dtype=np.int64))
            rows_arc.append(ids)
            rows_dist.append(d)
            rows_t.append(t)
            rows_fx.append(fx)
            rows_fy.append(fy)
        obs_idx = np.concatenate(rows_obs) if rows_obs else np.empty(0, dtype=np.int64)
        arc = np.concatenate(rows_arc) if rows_arc else np.empty(0, dtype=np.int64)
        dist = np.concatenate(rows_dist) if rows_dist else np.empty(0)
        t = np.concatenate(rows_t) if rows_t else np.empty(0)
        fx = np.concatenate(rows_fx) if rows_fx else np.empty(0)
        fy = np.concatenate(rows_fy) if rows_fy else np.empty(0)
        counts = np.bincount(obs_idx, minlength=n)
        rank = np.empty(len(obs_idx), dtype=np.int64)
        start = 0
        for i in range(n):
            c = int(counts[i])
            rank[start:start + c] = np.arange(c)
            start += c
        n_cand = counts[obs_idx] if len(obs_idx) else np.empty(0, dtype=np.int64)
        return CandidateSet(obs_idx=obs_idx, arc=arc, dist=dist, t=t, foot_x=fx, foot_y=fy,
                            rank=rank, n_cand=n_cand)


def track_points_xy(net: TripNetwork, trip: Trip) -> np.ndarray:
    x, y = to_local(trip.track[:, 0], trip.track[:, 1], net.frame)
    return np.column_stack([x, y])
