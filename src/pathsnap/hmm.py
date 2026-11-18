"""HMM route reconstruction with Viterbi decoding.

Hidden states are directed road arcs.  For every GPS fix we keep a small set of
candidate arcs.  The classifier supplies the emission likelihood of each
candidate; the transition score compares the road-network distance between two
candidates with the GPS displacement between the two fixes (Newson & Krumm).
Network distances are computed with a bounded, cached Dijkstra search and are
batched per time step, which keeps 1 Hz decoding interactive.
"""
from __future__ import annotations

import heapq

import numpy as np

from . import config
from .candidates import CandidateSet, TripNetwork

_EPS = 1e-12


class NetworkDistance:
    """Bounded shortest-path distances between nodes of one trip network."""

    def __init__(self, net: TripNetwork, max_search_m: float = config.HMM_MAX_SEARCH_M):
        self.net = net
        self.max_search = float(max_search_m)
        self.out = net.out_arcs()
        self.cache: dict[tuple[int, int], float] = {}
        self.fail_limit: dict[tuple[int, int], float] = {}
        self.arcs = net.trip.arcs
        self.len = net.len
        self.nx = net.node_x
        self.ny = net.node_y

    def _dijkstra(self, source: int, targets: set[int], limit: float) -> dict[int, float]:
        if not targets:
            return {}
        out = self.out
        arcs = self.arcs
        leng = self.len
        open_heap = [(0.0, source)]
        best = {source: 0.0}
        remaining = set(targets)
        found: dict[int, float] = {}
        if source in remaining:
            remaining.discard(source)
            found[source] = 0.0
        while open_heap and remaining:
            g, node = heapq.heappop(open_heap)
            if g > best.get(node, float("inf")):
                continue
            if node in remaining:
                # settle-on-pop: a distance is final only when the node is popped
                found[node] = g
                remaining.discard(node)
                if not remaining:
                    break
            if g > limit:
                break
            for a in out[node]:
                nxt = int(arcs[a, 1])
                ng = g + float(leng[a])
                if ng > limit or ng >= best.get(nxt, float("inf")):
                    continue
                best[nxt] = ng
                heapq.heappush(open_heap, (ng, nxt))
        return found

    def node_distances(self, sources, targets, limit: float) -> dict[tuple[int, int], float]:
        """Distances for requested (source, target) pairs, reusing the cache."""
        result: dict[tuple[int, int], float] = {}
        need: dict[int, set[int]] = {}
        for u, w in zip(sources, targets):
            u = int(u); w = int(w)
            if u == w:
                result[(u, w)] = 0.0
                continue
            key = (u, w)
            if key in self.cache:
                result[key] = self.cache[key]
                continue
            if key in self.fail_limit and self.fail_limit[key] >= limit:
                result[key] = float("inf")
                continue
            need.setdefault(u, set()).add(w)
        for u, ws in need.items():
            found = self._dijkstra(u, ws, limit)
            for w in ws:
                key = (u, w)
                if w in found:
                    self.cache[key] = found[w]
                    result[key] = found[w]
                else:
                    self.fail_limit[key] = max(self.fail_limit.get(key, 0.0), limit)
                    result[key] = float("inf")
        return result


def _step_log_transitions(nd: NetworkDistance, cand: CandidateSet,
                          prev_rows: np.ndarray, cur_rows: np.ndarray,
                          gps_d: float, beta: float) -> np.ndarray:
    """(K_prev, K_cur) matrix of log transition probabilities for one time step."""
    arcs = nd.arcs
    leng = nd.len
    a = cand.arc[prev_rows]; b = cand.arc[cur_rows]
    ta = cand.t[prev_rows][:, None]; tb = cand.t[cur_rows][None, :]
    la = leng[a][:, None]; lb = leng[b][None, :]
    base = (1.0 - ta) * la + tb * lb
    same = a[:, None] == b[None, :]
    same_net = np.abs(tb - ta) * la
    u = arcs[a, 1][:, None]
    w = arcs[b, 0][None, :]
    direct = u == w
    euclid = np.hypot(nd.nx[u] - nd.nx[w], nd.ny[u] - nd.ny[w])
    lower = base + euclid
    tol = gps_d + 4.5 * beta + 30.0
    plausible = (~same) & (~direct) & (lower <= tol)
    # pairs that cannot be reached within the tolerance are treated as
    # disconnected (infinite network distance) instead of using the lower bound
    net = np.where(same, same_net, base).astype(np.float64)
    net[(~same) & (~direct) & (~plausible)] = np.inf
    if plausible.any():
        ii, jj = np.where(plausible)
        uu = np.broadcast_to(u, net.shape)[ii, jj]
        ww = np.broadcast_to(w, net.shape)[ii, jj]
        limit = min(tol, nd.max_search)
        dist = nd.node_distances(uu, ww, limit)
        for k, (i, j, uu0, ww0) in enumerate(zip(ii, jj, uu, ww)):
            d = dist.get((int(uu0), int(ww0)), float("inf"))
            net[i, j] = base[i, j] + d if np.isfinite(d) else float("inf")
    finite = np.isfinite(net)
    logp = np.full(net.shape, -(tol / beta), dtype=np.float64)
    logp[finite] = -np.abs(net[finite] - gps_d) / beta
    logp = np.maximum(logp, -(tol / beta))
    return logp


def _state_rows(probs: np.ndarray, cand: CandidateSet, obs_i: int,
                max_states: int) -> np.ndarray:
    lo = int(np.searchsorted(cand.obs_idx, obs_i))
    hi = int(np.searchsorted(cand.obs_idx, obs_i, side="right"))
    if hi <= lo:
        return np.empty(0, dtype=np.int64)
    rows = np.arange(lo, hi)
    p = probs[lo:hi]
    if len(rows) > max_states:
        keep = np.argpartition(-p, max_states - 1)[:max_states]
        rows = rows[np.sort(keep)]
    return rows


def viterbi_decode(net: TripNetwork, cand: CandidateSet, probs: np.ndarray,
                   gps_xy: np.ndarray, arcs: np.ndarray,
                   beta: float = config.HMM_BETA_M,
                   emission_alpha: float = config.HMM_EMISSION_ALPHA,
                   max_states: int = config.HMM_MAX_STATES,
                   gap_free_s: float = config.HMM_GAP_FREE_S,
                   times: np.ndarray | None = None,
                   n_obs: int | None = None) -> np.ndarray:
    """Decode the most likely arc sequence; returns one arc id per observation."""
    n = int(n_obs if n_obs is not None else gps_xy.shape[0])
    pred = np.full(n, -1, dtype=np.int64)
    if len(cand) == 0 or n == 0:
        return pred
    nd = NetworkDistance(net)
    states: list[np.ndarray] = []
    log_em: list[np.ndarray] = []
    for i in range(n):
        rows = _state_rows(probs, cand, i, max_states)
        states.append(rows)
        if len(rows) == 0:
            log_em.append(np.empty(0))
            continue
        p = np.clip(probs[rows], _EPS, None)
        p = p / p.sum()
        log_em.append(emission_alpha * np.log(np.clip(p, _EPS, 1.0)))

    log_prob = np.full((n, max_states), -np.inf)
    back = np.full((n, max_states), -1, dtype=np.int64)
    for i in range(n):
        k = len(states[i])
        if k == 0:
            continue
        if i == 0 or len(states[i - 1]) == 0:
            log_prob[i, :k] = log_em[i]
            continue
        dt = 1.0 if times is None else float(max(times[i] - times[i - 1], 0.1))
        gps_d = float(np.hypot(gps_xy[i, 0] - gps_xy[i - 1, 0],
                               gps_xy[i, 1] - gps_xy[i - 1, 1]))
        if dt > gap_free_s:
            trans = np.zeros((len(states[i - 1]), k))
        else:
            trans = _step_log_transitions(nd, cand, states[i - 1], states[i], gps_d, beta)
        mat = log_prob[i - 1, :len(states[i - 1])][:, None] + trans
        jbest = np.argmax(mat, axis=0)
        log_prob[i, :k] = log_em[i] + mat[jbest, np.arange(k)]
        back[i, :k] = jbest

    last = -1
    for i in range(n - 1, -1, -1):
        if len(states[i]):
            last = i
            break
    if last < 0:
        return pred
    sel = int(np.argmax(log_prob[last, :len(states[last])]))
    for i in range(last, -1, -1):
        if len(states[i]) == 0:
            continue
        pred[i] = int(cand.arc[states[i][sel]])
        if i > 0:
            j = int(back[i, sel])
            sel = j if j >= 0 else 0
    for i in range(n):
        if pred[i] < 0:
            if i > 0:
                pred[i] = pred[i - 1]
            else:
                future = np.where(pred >= 0)[0]
                pred[i] = pred[future[0]] if len(future) else -1
    return pred
