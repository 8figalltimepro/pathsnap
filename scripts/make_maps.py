"""Generate interactive Folium maps comparing GPS, road network, reference route
and the two predicted routes (classifier-only vs HMM) for development trips.

Run after scripts/evaluate_final.py (needs artifacts/results/final_predictions.npz).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import folium
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import config
from pathsnap.candidates import TripNetwork, track_points_xy
from pathsnap.geo import to_lonlat
from pathsnap.io import load_trip

RESULTS = config.ARTIFACTS_DIR / "results"
OUT = ROOT / "reports" / "maps"
OUT.mkdir(parents=True, exist_ok=True)


def sample(seq, step):
    return seq[::step] if step > 1 else seq


def build_map(trip_id: str, partition: str, predictions: dict, max_points: int = 6000):
    trip = load_trip(trip_id, partition)
    net = TripNetwork(trip)
    obs_xy = track_points_xy(net, trip)
    gps_lon, gps_lat = to_lonlat(obs_xy[:, 0], obs_xy[:, 1], net.frame)
    center = [float(np.mean(gps_lat)), float(np.mean(gps_lon))]
    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap",
                      control_scale=True, prefer_canvas=True)

    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([net.node_x, net.node_y]))
    hits = tree.query_ball_point(obs_xy[:: max(1, len(obs_xy) // 3000)], r=180.0)
    keep = np.zeros(len(net.node_x), dtype=bool)
    for lst in hits:
        keep[np.asarray(lst, dtype=np.int64)] = True
    arcs = trip.arcs
    arc_keep = keep[arcs[:, 0]] & keep[arcs[:, 1]]
    node_ids = np.where(keep)[0]
    remap = np.full(len(net.node_x), -1, dtype=np.int64)
    remap[node_ids] = np.arange(len(node_ids))

    net_group = folium.FeatureGroup(name="Road network", show=True)
    for a in np.where(arc_keep)[0]:
        u, v = remap[arcs[a, 0]], remap[arcs[a, 1]]
        lonlat = to_lonlat([net.node_x[node_ids[u]], net.node_x[node_ids[v]]],
                           [net.node_y[node_ids[u]], net.node_y[node_ids[v]]], net.frame)
        folium.PolyLine([[lonlat[1][0], lonlat[0][0]], [lonlat[1][1], lonlat[0][1]]],
                        color="#666666", weight=1.2, opacity=0.55,
                        tooltip=f"arc {a}").add_to(net_group)
    net_group.add_to(fmap)

    ref_group = folium.FeatureGroup(name="Reference route (ground truth)", show=True)
    if trip.route is not None:
        for a in trip.route:
            a = int(a)
            lonlat = to_lonlat([net.ax[a], net.bx[a]], [net.ay[a], net.by[a]], net.frame)
            folium.PolyLine([[lonlat[1][0], lonlat[0][0]], [lonlat[1][1], lonlat[0][1]]],
                            color="#2ca02c", weight=4, opacity=0.85).add_to(ref_group)
    ref_group.add_to(fmap)

    gps_group = folium.FeatureGroup(name="GPS fixes", show=True)
    step = max(1, len(gps_lon) // max_points)
    for lon, lat in zip(sample(gps_lon, step), sample(gps_lat, step)):
        folium.CircleMarker([float(lat), float(lon)], radius=1.6, color="#1f77b4",
                            fill=True, fill_opacity=0.7, weight=0).add_to(gps_group)
    gps_group.add_to(fmap)

    def add_pred(group_name, key, color, dashed=False):
        pred = predictions.get(key)
        if pred is None:
            return
        group = folium.FeatureGroup(name=group_name, show=True)
        chains = []
        cur = []
        prev = None
        for a in pred:
            a = int(a)
            if a < 0:
                continue
            if prev is not None and a != prev and int(arcs[a, 0]) != int(arcs[prev, 1]):
                chains.append(cur); cur = []
            if prev is None or a != prev:
                cur.append(a)
            prev = a
        if cur:
            chains.append(cur)
        for chain in chains:
            if not chain:
                continue
            node_seq = [int(arcs[chain[0], 0])] + [int(arcs[a, 1]) for a in chain]
            lon, lat = to_lonlat(net.node_x[node_seq], net.node_y[node_seq], net.frame)
            style = {"color": color, "weight": 3 if dashed else 5, "opacity": 0.9}
            if dashed:
                style["dash_array"] = "8 6"
            folium.PolyLine([[float(b), float(a)] for a, b in zip(lon, lat)],
                            tooltip=group_name, **style).add_to(group)
        group.add_to(fmap)

    add_pred("Classifier-only route", f"{trip_id}_clf", "#ff7f0e", dashed=True)
    add_pred("HMM predicted route", f"{trip_id}_hmm", "#d62728")

    legend = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.3); font-size: 13px; line-height: 1.6;">
      <b>PathSnap layers</b><br>
      <span style="color:#1f77b4;">&#9679;</span> GPS fixes<br>
      <span style="color:#666;">&#9473;</span> Road network<br>
      <span style="color:#2ca02c;">&#9473;</span> Reference route (ground truth)<br>
      <span style="color:#ff7f0e;">&#9473;</span> Classifier-only route<br>
      <span style="color:#d62728;">&#9473;</span> HMM + Viterbi route
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[float(np.min(gps_lat)) - 0.01, float(np.min(gps_lon)) - 0.01],
                     [float(np.max(gps_lat)) + 0.01, float(np.max(gps_lon)) + 0.01]])
    return fmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="+", default=["00000000", "00000040", "00000091"])
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    preds_path = RESULTS / "final_predictions.npz"
    if not preds_path.exists():
        raise SystemExit("run scripts/evaluate_final.py first")
    z = np.load(preds_path)
    predictions = {k: z[k] for k in z.files}
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for tid in args.trips:
        fmap = build_map(tid, args.partition, predictions)
        path = out_dir / f"predicted_{tid}.html"
        fmap.save(str(path))
        written.append(str(path))
        print("wrote", path)
    (RESULTS / "map_index.json").write_text(json.dumps(
        {"maps": written, "partition": args.partition}, indent=2))


if __name__ == "__main__":
    main()
