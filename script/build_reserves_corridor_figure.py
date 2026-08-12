#!/usr/bin/env python3
"""
build_reserves_corridor_figure.py
─────────────────────────────────
Reviewer 4 asked to see the distribution of reserves and corridors, and how the
study supports reserve design. Two panels:

  (A) Regional context — Important Bird Areas / Key Biodiversity Areas within
      ~60 km, with the UCPS AOI positioned between them and edge-to-edge
      distances to the three nearest sites annotated.
  (B) AOI detail — occupied cells by CPI tier, with unsurveyed corridor cells
      classified into Tier 1-3, on the Sentinel-2 basemap.

Writes figures/fig_reserves_corridors.{pdf,png}
"""

import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import Polygon
from shapely.ops import nearest_points

import h3


def _reeps_base() -> Path:
    env = os.environ.get("REEPS_BASE")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for cand in (here.parent, *here.parents):
        if (cand / "REEPS_Master_Database.gpkg").exists():
            return cand
    return here.parent


BASE = _reeps_base()
sys.path.insert(0, str(BASE / "script"))

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 0.6,
})

TIER_COLOR = {"CRITICAL": "#B71C1C", "HIGH": "#EF6C00",
              "MEDIUM": "#FBC02D", "LOW": "#C5E1A5"}
CORR_COLOR = {"Tier 1": "#4A148C", "Tier 2": "#7B1FA2", "Tier 3": "#BA68C8"}
UTM = 32748


def hex_poly(cell: str) -> Polygon:
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def main() -> None:
    aoi = gpd.read_file(BASE / "aoi.gpkg").to_crs(UTM)
    iba = gpd.read_file(
        BASE / "IBA/Important_Bird_Area_Indonesia_Mar2018.shp").to_crs(UTM)
    pri = pd.read_csv(BASE / "priority_index.csv")
    step = pd.read_csv(BASE / "stepping_stones.csv")

    a_geom = aoi.geometry.union_all()
    iba["d_km"] = (iba.geometry.distance(a_geom) / 1000).round(1)
    near = iba[iba["d_km"] < 60].copy().sort_values("d_km")
    print(f"IBAs within 60 km: {len(near)}")

    fig = plt.figure(figsize=(13.6, 6.2))
    axA = fig.add_axes([0.035, 0.09, 0.44, 0.84])
    axB = fig.add_axes([0.525, 0.09, 0.44, 0.84])

    # ── Panel A ─────────────────────────────────────────────────────────────
    near.plot(ax=axA, facecolor="#2E7D32", edgecolor="#1B5E20",
              alpha=0.45, linewidth=0.6, zorder=2)
    aoi.boundary.plot(ax=axA, color="#E65100", linewidth=2.4, zorder=5)

    for _, r in near.head(3).iterrows():
        c = r.geometry.centroid
        axA.annotate(f"{r['NatName']}\n{r['d_km']:.0f} km",
                     xy=(c.x, c.y), ha="center", va="center", fontsize=7.5,
                     color="#1B5E20", weight="bold", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white",
                               ec="#1B5E20", lw=0.5, alpha=0.85))
        # draw the actual shortest AOI-to-reserve link, so the line matches the
        # edge-to-edge distance quoted in the label
        p_aoi, p_res = nearest_points(a_geom, r.geometry)
        axA.plot([p_aoi.x, p_res.x], [p_aoi.y, p_res.y],
                 color="#424242", lw=0.9, ls="--", zorder=3)
        axA.plot([p_aoi.x, p_res.x], [p_aoi.y, p_res.y], "o",
                 ms=2.4, color="#424242", zorder=4)

    ac = a_geom.centroid
    axA.annotate("UCPS AOI", xy=(ac.x, ac.y), xytext=(ac.x, ac.y - 11000),
                 ha="center", fontsize=9, weight="bold", color="#E65100",
                 arrowprops=dict(arrowstyle="->", color="#E65100", lw=1.2),
                 zorder=7)

    minx, miny, maxx, maxy = near.total_bounds
    axA.set_xlim(minx - 4000, maxx + 4000)
    axA.set_ylim(miny - 4000, maxy + 4000)
    axA.set_title("(a)  Regional protected-area context", loc="left", pad=6,
                  weight="bold")
    axA.set_xticks([]); axA.set_yticks([])
    for s in axA.spines.values():
        s.set_edgecolor("#9E9E9E")

    # scale bar
    x0, y0 = axA.get_xlim()[0] + 5000, axA.get_ylim()[0] + 5000
    axA.plot([x0, x0 + 20000], [y0, y0], color="black", lw=2.5, zorder=8)
    axA.text(x0 + 10000, y0 + 1800, "20 km", ha="center", fontsize=8, zorder=8)
    axA.annotate("N", xy=(axA.get_xlim()[1] - 6000, axA.get_ylim()[1] - 6000),
                 ha="center", fontsize=11, weight="bold",
                 xytext=(axA.get_xlim()[1] - 6000, axA.get_ylim()[1] - 16000),
                 arrowprops=dict(arrowstyle="->", lw=1.4, color="black"))

    axA.legend(handles=[
        mpatches.Patch(fc="#2E7D32", ec="#1B5E20", alpha=0.45,
                       label="IBA / KBA"),
        Line2D([], [], color="#E65100", lw=2.4, label="UCPS AOI"),
        Line2D([], [], color="#616161", lw=0.7, ls="--",
               label="shortest AOI-reserve link"),
    ], loc="upper left", frameon=True, framealpha=0.9)

    # ── Panel B ─────────────────────────────────────────────────────────────
    try:
        from basemap_utils import S2_RGBA, S2_EXTENT
        have_bm = True
    except Exception as e:                                  # pragma: no cover
        print(f"  basemap unavailable ({e}); drawing without it")
        have_bm = False

    occ_g = gpd.GeoDataFrame(
        pri, geometry=[hex_poly(c) for c in pri["h3_index"]], crs=4326)
    corr = step[step["corridor_tier"] != "None"].copy()
    corr_g = gpd.GeoDataFrame(
        corr, geometry=[hex_poly(c) for c in corr["h3_index"]], crs=4326)

    if have_bm:
        axB.imshow(S2_RGBA, extent=S2_EXTENT, origin="upper", zorder=0)

    for tier, col in TIER_COLOR.items():
        sub = occ_g[occ_g["Priority_Tier"] == tier]
        if len(sub):
            sub.plot(ax=axB, facecolor=col, edgecolor="#37474F",
                     linewidth=0.4, alpha=0.78, zorder=3)
    for tier, col in CORR_COLOR.items():
        sub = corr_g[corr_g["corridor_tier"] == tier]
        if len(sub):
            sub.plot(ax=axB, facecolor="none", edgecolor=col,
                     linewidth=1.6, zorder=4)

    aoi.to_crs(4326).boundary.plot(ax=axB, color="#E65100", linewidth=1.8,
                                   zorder=5)

    b = occ_g.total_bounds
    padx = (b[2] - b[0]) * 0.16
    pady = (b[3] - b[1]) * 0.16
    axB.set_xlim(b[0] - padx, b[2] + padx)
    axB.set_ylim(b[1] - pady, b[3] + pady)
    axB.set_title("(b)  Conservation priority and corridor cells",
                  loc="left", pad=6, weight="bold")
    axB.set_xticks([]); axB.set_yticks([])
    for s in axB.spines.values():
        s.set_edgecolor("#9E9E9E")

    counts = pri["Priority_Tier"].value_counts()
    ccounts = corr["corridor_tier"].value_counts()
    handles = [mpatches.Patch(fc=c, ec="#37474F", alpha=0.78,
                              label=f"CPI {t} ({counts.get(t, 0)})")
               for t, c in TIER_COLOR.items()]
    handles += [mpatches.Patch(fc="none", ec=c,
                               label=f"Corridor {t} ({ccounts.get(t, 0)})")
                for t, c in CORR_COLOR.items()]
    axB.legend(handles=handles, loc="upper left", frameon=True,
               framealpha=0.9, ncol=2)

    nearest = near.iloc[0]
    fig.text(0.5, 0.015,
             f"UCPS AOI in its regional protected-area context. Nearest site: "
             f"{nearest['NatName']} ({nearest['d_km']:.1f} km from the AOI edge). "
             f"Corridor tiers follow adjacency to CRITICAL (Tier 1) and HIGH "
             f"(Tier 2) priority cells; Tier 3 cells lie on shortest paths "
             f"between high-priority cells.",
             ha="center", fontsize=7.5, style="italic", color="#424242")

    out = BASE / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"fig_reserves_corridors.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  wrote {p.name} ({p.stat().st_size/1e6:.1f} MB)")
    plt.close(fig)

    print("\nnearest IBAs (edge-to-edge):")
    for _, r in near.head(5).iterrows():
        print(f"  {str(r['NatName'])[:36]:36s} {r['d_km']:5.1f} km")


if __name__ == "__main__":
    main()
