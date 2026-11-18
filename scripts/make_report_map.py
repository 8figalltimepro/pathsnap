"""Build the single Folium map used in the minimal report: one trip with the raw
GPS fixes, the local road network, the reference route and the predicted
HMM/Viterbi route. Interactive HTML is committed; a screenshot goes into the PDF.
"""
from __future__ import annotations

import sys
from pathlib import Path

import folium
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import config
from pathsnap.candidates import TripNetwork, track_points_xy
from pathsnap.geo import to_lonlat
from pathsnap.io import load_trip

RESULTS = config.ARTIFACTS_DIR / "results"
OUT = ROOT / "reports" / "maps"
TRIP = "00000052"
PARTITION = "student_test"


def main():
    trip = load_trip(TRIP, PARTITION)
    net = TripNetwork(trip)
    obs_xy = track_points_xy(net, trip)
    lon, lat = to_lonlat(obs_xy[:, 0], obs_xy[:, 1], net.frame)
    pred = np.load(RESULTS / "final_predictions.npz")[f"{TRIP}_hmm"]
    arcs = trip.arcs

    fmap = folium.Map(location=[float(np.mean(lat)), float(np.mean(lon))],
                      zoom_start=13, tiles="OpenStreetMap", control_scale=True,
                      prefer_canvas=False)
    # road network near the trajectory
    tree = cKDTree(np.column_stack([net.node_x, net.node_y]))
    hits = tree.query_ball_point(obs_xy, r=180.0)
    keep = np.zeros(len(net.node_x), dtype=bool)
    for h in hits:
        keep[np.asarray(h, dtype=np.int64)] = True
    net_group = folium.FeatureGroup(name="Road network")
    for aIdx in np.where(keep[arcs[:, 0]] & keep[arcs[:, 1]])[0]:
        ll = to_lonlat([net.ax[aIdx], net.bx[aIdx]], [net.ay[aIdx], net.by[aIdx]], net.frame)
        folium.PolyLine([[ll[1][0], ll[0][0]], [ll[1][1], ll[0][1]]],
                        color="#8F8F8F", weight=1.2, opacity=0.75).add_to(net_group)
    net_group.add_to(fmap)

    # GPS fixes first (under the routes) so the route lines stay continuous
    gps_group = folium.FeatureGroup(name="GPS fixes")
    step = max(1, len(lon) // 3000)
    for i in range(0, len(lon), step):
        folium.CircleMarker([float(lat[i]), float(lon[i])], radius=1.7, color="#ffffff",
                            weight=0.4, fill=True, fill_color="#6b7280",
                            fill_opacity=0.8).add_to(gps_group)
    gps_group.add_to(fmap)

    def chains_from_arcs(arc_list):
        chains, cur = [], []
        for a in arc_list:
            a = int(a)
            if a < 0:
                if cur:
                    chains.append(cur); cur = []
                continue
            if cur:
                prev = cur[-1]
                if a != prev and int(arcs[a, 0]) != int(arcs[prev, 1]):
                    chains.append(cur); cur = []
                elif a == prev:
                    continue
            cur.append(a)
        if cur:
            chains.append(cur)
        out = []
        for ch in chains:
            nodes = [int(arcs[ch[0], 0])] + [int(arcs[x, 1]) for x in ch]
            out.append(nodes)
        return out

    def draw_route(arc_list, color, weight, name):
        grp = folium.FeatureGroup(name=name)
        for nodes in chains_from_arcs(arc_list):
            ll = to_lonlat(net.node_x[nodes], net.node_y[nodes], net.frame)
            folium.PolyLine([[float(b), float(a)] for a, b in zip(ll[0], ll[1])],
                            color=color, weight=weight, opacity=0.95).add_to(grp)
        grp.add_to(fmap)

    draw_route(trip.route, "#0072F5", 6.0, "Reference route")
    draw_route(pred, "#45A557", 3.8, "HMM/Viterbi route")

    legend = (
        '<div style="position: fixed; top: 76px; right: 22px; z-index: 9999; '
        'background:#ffffff; border:1px solid #e5e5e5; border-radius:6px; '
        'padding:10px 14px; font:18px/1.65 -apple-system,Geist,Segoe UI,sans-serif; '
        'color:#111; box-shadow:0 1px 2px rgba(0,0,0,.06);">'
        '<span style="color:#8F8F8F;">&#9473;</span> road network&nbsp;&nbsp;'
        '<span style="color:#0072F5;">&#9473;</span> reference route<br>'
        '<span style="color:#45A557;">&#9473;</span> HMM/Viterbi route&nbsp;&nbsp;'
        '<span style="color:#6b7280;">&#9679;</span> GPS fix'
        "</div>"
    )
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(fmap)
    pad_lat = max(0.002, 0.03 * float(np.ptp(lat)))
    pad_lon = max(0.002, 0.03 * float(np.ptp(lon)))
    fmap.fit_bounds([[float(lat.min() - pad_lat), float(lon.min() - pad_lon)],
                     [float(lat.max() + pad_lat), float(lon.max() + pad_lon)]])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "report_map.html"
    fmap.save(str(path))
    print("wrote", path, "trip", TRIP, "fixes", len(lon))


if __name__ == "__main__":
    main()
