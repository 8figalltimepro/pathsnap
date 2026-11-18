"""Geographical helpers.

All calculations use a per-trip local metric frame (equirectangular around the
trip centre) which is accurate to well under a metre over the extents in this
dataset, plus haversine distances for anything reported in absolute terms.
"""
from __future__ import annotations

import numpy as np

R_EARTH_M = 6371008.8


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres, vectorised over numpy arrays or floats."""
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_EARTH_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def make_frame(lon, lat):
    """Return (lon0, lat0) for an equirectangular local frame."""
    return float(np.mean(lon)), float(np.mean(lat))


def to_local(lon, lat, frame):
    """Project lon/lat to local metres: x east, y north."""
    lon0, lat0 = frame
    x = np.radians(np.asarray(lon, dtype=float) - lon0) * R_EARTH_M * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat, dtype=float) - lat0) * R_EARTH_M
    return x, y


def to_lonlat(x, y, frame):
    """Inverse of to_local."""
    lon0, lat0 = frame
    lon = lon0 + np.degrees(np.asarray(x, dtype=float) / (R_EARTH_M * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees(np.asarray(y, dtype=float) / R_EARTH_M)
    return lon, lat


def bearing_rad(lon1, lat1, lon2, lat2):
    """Bearing in radians (-pi..pi] from point 1 to point 2."""
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(x, y)


def vec_bearing_rad(dx, dy):
    """Bearing of a local metric vector."""
    return np.arctan2(dx, dy)


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def angle_diff(a, b):
    """Signed smallest difference a - b, wrapped to (-pi, pi]."""
    return wrap_angle(np.asarray(a) - np.asarray(b))


def point_to_segment(px, py, ax, ay, bx, by):
    """Vectorised point-to-segment distance.

    px/py can be (n,) or (n,1); ax/ay/bx/by can be (m,) or (1,m). Broadcasting
    produces an (n, m) result when both are used together.
    Returns dist, t (projection fraction clipped to [0,1]), foot_x, foot_y.
    """
    ax = np.asarray(ax, dtype=float)
    ay = np.asarray(ay, dtype=float)
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    dx = bx - ax
    dy = by - ay
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    seg2 = dx * dx + dy * dy
    safe = np.where(seg2 > 1e-12, seg2, 1.0)
    t = ((px - ax) * dx + (py - ay) * dy) / safe
    t = np.clip(t, 0.0, 1.0)
    fx = ax + t * dx
    fy = ay + t * dy
    dist = np.hypot(px - fx, py - fy)
    return dist, t, fx, fy


def project_on_polyline(px, py, verts, chunk=1024):
    """Project points on a polyline given as (m+1, 2) vertices.

    Returns best segment index, distance and along-polyline distance in metres.
    """
    verts = np.asarray(verts, dtype=float)
    seg_a = verts[:-1]
    seg_b = verts[1:]
    seg_len = np.hypot(seg_b[:, 0] - seg_a[:, 0], seg_b[:, 1] - seg_a[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    pix = np.asarray(px, dtype=float)
    piy = np.asarray(py, dtype=float)
    n = len(pix)
    best_idx = np.empty(n, dtype=np.int32)
    best_dist = np.empty(n, dtype=float)
    best_along = np.empty(n, dtype=float)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d, t, _, _ = point_to_segment(
            pix[start:stop, None], piy[start:stop, None],
            seg_a[None, :, 0], seg_a[None, :, 1], seg_b[None, :, 0], seg_b[None, :, 1])
        j = np.argmin(d, axis=1)
        rows = np.arange(stop - start)
        best_idx[start:stop] = j
        best_dist[start:stop] = d[rows, j]
        best_along[start:stop] = cum[j] + t[rows, j] * seg_len[j]
    return best_idx, best_dist, best_along


def project_on_arcs(px, py, ax, ay, bx, by):
    """Point-to-segment distance for a small set of candidate arcs.

    Returns distance, projection fraction, foot coordinates.
    """
    d, t, fx, fy = point_to_segment(px, py, ax, ay, bx, by)
    return d, t, fx, fy
