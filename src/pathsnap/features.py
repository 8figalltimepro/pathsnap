"""Feature engineering for candidate arcs.

Each row is one (GPS fix, candidate road arc) pair.  The same feature builder
is used for training and inference, so the model always sees identical inputs.
"""
from __future__ import annotations

import numpy as np

from .candidates import CandidateSet, TripNetwork
from .geo import angle_diff, haversine_m, vec_bearing_rad, wrap_angle
from .io import Trip

FEATURE_NAMES = [
    "perp_dist_m",        # distance from the fix to the arc segment
    "log_perp_dist",
    "dist_rank",          # 0 = closest candidate for this fix
    "dist_gap_best_m",    # extra distance versus the closest candidate
    "n_cand",
    "rel_dist",           # perp / (best + 1)
    "heading_diff_abs",   # |bearing difference| movement vs arc, radians
    "cos_heading_diff",
    "arc_len_m",
    "log_arc_len",
    "t_along",            # projection fraction along the arc
    "dist_to_from_node_m",
    "dist_to_to_node_m",
    "step_dist_m",        # distance between neighbouring fixes
    "speed_mps",
    "log_dt",
    "dist_prev_fix_m",    # distance from previous fix to this arc
    "dist_next_fix_m",
    "foot_step_ratio",    # foot displacement vs GPS step displacement
    "hop_prev",           # network hops from previous fix candidates (0-4)
    "hop_next",           # network hops to next fix candidates (0-4)
    "same_as_prev_closest",
    "same_as_next_closest",
    "from_degree",
    "to_degree",
    "log_from_degree",
]

N_FEATURES = len(FEATURE_NAMES)
_MAX_DEGREE_ENUM = 60
_SENTINEL_DIST = 500.0


def _degree_lists(net: TripNetwork):
    out = net.out_arcs()
    deg = np.zeros(len(net.node_x), dtype=np.float32)
    for node, lst in enumerate(out):
        deg[node] = len(lst)
    return deg


def _min_hops(net: TripNetwork, prev_arcs: np.ndarray, cur_arcs: np.ndarray,
              arcs: np.ndarray, out_lists, max_hops: int = 3) -> np.ndarray:
    """For each current arc, minimum number of network hops from any previous arc."""
    res = np.full(len(cur_arcs), max_hops + 1, dtype=np.float32)
    if len(prev_arcs) == 0 or len(cur_arcs) == 0:
        return res
    prev_set = set(int(a) for a in prev_arcs)
    cur_to = arcs[cur_arcs, 1]
    prev_to = arcs[prev_arcs, 1]
    cur_from = arcs[cur_arcs, 0]
    # 0 hops: same arc
    for k, a in enumerate(cur_arcs):
        if int(a) in prev_set:
            res[k] = 0
    if res.min() == 0:
        pass
    # 1 hop
    prev_to_set = set(int(n) for n in prev_to)
    for k in range(len(cur_arcs)):
        if res[k] > 1 and int(cur_from[k]) in prev_to_set:
            res[k] = 1
    # 2 hops
    if res.max() > 2:
        two_hop_nodes = set()
        for a in prev_arcs:
            for c in out_lists[int(arcs[a, 1])][:_MAX_DEGREE_ENUM]:
                two_hop_nodes.add(int(arcs[c, 1]))
        for k in range(len(cur_arcs)):
            if res[k] > 2 and int(cur_from[k]) in two_hop_nodes:
                res[k] = 2
    # 3 hops (bounded breadth-first expansion)
    if res.max() > 3:
        three_hop_nodes = set()
        for a in prev_arcs:
            for c in out_lists[int(arcs[a, 1])][:_MAX_DEGREE_ENUM]:
                for d in out_lists[int(arcs[c, 1])][:_MAX_DEGREE_ENUM]:
                    three_hop_nodes.add(int(arcs[d, 1]))
        for k in range(len(cur_arcs)):
            if res[k] > 3 and int(cur_from[k]) in three_hop_nodes:
                res[k] = 3
    return res


def build_features(net: TripNetwork, trip: Trip, cand: CandidateSet) -> dict:
    """Return a dict with X (float32), y (labels or -1), obs_idx, arc ids."""
    obs_xy = np.column_stack([net.node_x[0] * 0 + 0.0])  # placeholder replaced below
    # track points in the local metric frame
    from .geo import to_local
    px, py = to_local(trip.track[:, 0], trip.track[:, 1], net.frame)
    n = len(trip.track)
    times = trip.track[:, 2]
    dt_prev = np.empty(n, dtype=np.float64)
    dt_prev[0] = 1.0
    dt_prev[1:] = np.clip(np.diff(times), 0.05, None)
    # smoothed movement heading: from previous to next fix
    sx = np.empty(n); sy = np.empty(n)
    sx[0] = px[1] - px[0] if n > 1 else 0.0; sy[0] = py[1] - py[0] if n > 1 else 0.0
    sx[-1] = px[-1] - px[-2] if n > 1 else 0.0; sy[-1] = py[-1] - py[-2] if n > 1 else 0.0
    if n > 2:
        sx[1:-1] = px[2:] - px[:-2]; sy[1:-1] = py[2:] - py[:-2]
    heading = vec_bearing_rad(sx, sy)
    step_dist = np.hypot(sx, sy)
    speed = step_dist / dt_prev
    prev_dx = np.empty(n); prev_dy = np.empty(n)
    prev_dx[0] = 0.0; prev_dy[0] = 0.0
    prev_dx[1:] = px[1:] - px[:-1]; prev_dy[1:] = py[1:] - py[:-1]

    arcs = trip.arcs
    ax_all = net.ax; ay_all = net.ay; bx_all = net.bx; by_all = net.by
    arc_len_all = net.len; bearing_all = net.bearing
    deg = _degree_lists(net)
    out_lists = net.out_arcs()

    row_obs = cand.obs_idx
    row_arc = cand.arc
    r_ax = ax_all[row_arc]; r_ay = ay_all[row_arc]
    r_bx = bx_all[row_arc]; r_by = by_all[row_arc]
    # distance from previous / next fix to each row's arc (vectorised)
    def _dist_point_arc(qx, qy):
        dx = r_bx - r_ax; dy = r_by - r_ay
        seg2 = dx * dx + dy * dy
        safe = np.where(seg2 > 1e-12, seg2, 1.0)
        t = ((qx - r_ax) * dx + (qy - r_ay) * dy) / safe
        t = np.clip(t, 0.0, 1.0)
        fx = r_ax + t * dx; fy = r_ay + t * dy
        return np.hypot(qx - fx, qy - fy), fx, fy

    qx_prev = px[np.maximum(row_obs - 1, 0)]
    qy_prev = py[np.maximum(row_obs - 1, 0)]
    qx_next = px[np.minimum(row_obs + 1, n - 1)]
    qy_next = py[np.minimum(row_obs + 1, n - 1)]
    dist_prev, _, _ = _dist_point_arc(qx_prev, qy_prev)
    dist_next, _, _ = _dist_point_arc(qx_next, qy_next)
    # stop the sentinel rows (first/last fix) from dominating
    dist_prev[row_obs == 0] = _SENTINEL_DIST
    dist_next[row_obs == n - 1] = _SENTINEL_DIST

    hd = angle_diff(bearing_all[row_arc], heading[row_obs])
    # foot points of the candidate rows
    dx = r_bx - r_ax; dy = r_by - r_ay
    seg2 = dx * dx + dy * dy
    safe = np.where(seg2 > 1e-12, seg2, 1.0)
    t = ((px[row_obs] - r_ax) * dx + (py[row_obs] - r_ay) * dy) / safe
    t = np.clip(t, 0.0, 1.0)
    fx = r_ax + t * dx; fy = r_ay + t * dy

    # minimal network hops from previous / next observation's candidate set
    hop_prev = np.full(len(row_arc), 4.0, dtype=np.float32)
    hop_next = np.full(len(row_arc), 4.0, dtype=np.float32)
    biggest = int(np.bincount(row_obs, minlength=n).max()) if len(row_obs) else 0
    if biggest > 0:
        starts = np.searchsorted(row_obs, np.arange(n))
        ends = np.searchsorted(row_obs, np.arange(n), side="right")
    for i in range(n):
        if biggest == 0 or ends[i] <= starts[i]:
            continue
        s, e = starts[i], ends[i]
        cur = row_arc[s:e]
        if i > 0 and ends[i - 1] > starts[i - 1]:
            hp = _min_hops(net, row_arc[starts[i - 1]:ends[i - 1]], cur, arcs, out_lists)
            hop_prev[s:e] = np.minimum(hop_prev[s:e], hp)
        if i < n - 1 and ends[i + 1] > starts[i + 1]:
            hn = _min_hops(net, row_arc[starts[i + 1]:ends[i + 1]], cur, arcs, out_lists)
            hop_next[s:e] = np.minimum(hop_next[s:e], hn)

    same_prev = np.zeros(len(row_arc), dtype=np.float32)
    same_next = np.zeros(len(row_arc), dtype=np.float32)
    if biggest > 0:
        for i in range(n):
            s, e = starts[i], ends[i]
            if e <= s:
                continue
            if i > 0 and ends[i - 1] > starts[i - 1]:
                same_prev[s:e] = (row_arc[s:e] == row_arc[starts[i - 1]]).astype(np.float32)
            if i < n - 1 and ends[i + 1] > starts[i + 1]:
                same_next[s:e] = (row_arc[s:e] == row_arc[starts[i + 1]]).astype(np.float32)

    # foot displacement relative to GPS displacement
    foot_dx = fx - px[np.maximum(row_obs - 1, 0)]
    foot_dy = fy - py[np.maximum(row_obs - 1, 0)]
    foot_step = np.hypot(foot_dx, foot_dy)
    step = np.maximum(step_dist[row_obs], 1e-3)
    foot_step_ratio = np.clip(foot_step / step, 0.0, 10.0)
    from_deg = deg[arcs[row_arc, 0]]
    to_deg = deg[arcs[row_arc, 1]]

    # per-observation aggregates: gap to the closest candidate and relative distance
    counts = np.bincount(row_obs, minlength=n) if len(row_obs) else np.zeros(n, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    best = np.full(n, np.inf)
    if len(row_obs):
        np.minimum.at(best, row_obs, cand.dist)
    best_rep = best[row_obs] if len(row_obs) else np.empty(0)

    X = np.column_stack([
        cand.dist,
        np.log1p(cand.dist),
        cand.rank.astype(np.float32),
        cand.dist - best_rep,
        cand.n_cand.astype(np.float32),
        cand.dist / (best_rep + 1.0),
        np.abs(hd),
        np.cos(hd),
        arc_len_all[row_arc],
        np.log1p(arc_len_all[row_arc]),
        cand.t,
        np.hypot(px[row_obs] - r_ax, py[row_obs] - r_ay),
        np.hypot(px[row_obs] - r_bx, py[row_obs] - r_by),
        step_dist[row_obs],
        speed[row_obs],
        np.log1p(dt_prev[row_obs]),
        dist_prev,
        dist_next,
        foot_step_ratio,
        hop_prev,
        hop_next,
        same_prev,
        same_next,
        from_deg,
        to_deg,
        np.log1p(from_deg),
    ]).astype(np.float32)
    return {"X": X, "obs_idx": row_obs, "arc": row_arc,
            "n_obs": n, "feature_names": FEATURE_NAMES}
