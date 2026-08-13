#!/usr/bin/env python3
"""Supplementary Figure S2 — Local Indicators of Spatial Association (LISA).

Written in revision to replace an inconsistent set of LISA outputs. Three different
tallies were in circulation: the manuscript reported 6 High-High / 3 Low-Low / 1
High-Low / 1 Low-High / 27 not significant; the saved lisa_results.csv held
6 / 1 / 1 / 1 / 29; and the figure's own legend claimed 11 significant cells against
the 9 in that file. The figure predated the coordinate correction.

The instability is real rather than clerical. Local Moran's I is assessed by
conditional permutation, and at n = 38 with several cells sitting at p = 0.04-0.07
the Low-Low and not-significant counts move with the random seed. The High-High
cluster, which is what the manuscript's argument rests on, is stable.

This script therefore fixes the seed, writes it into the outputs, and reports the
seed sensitivity explicitly so the numbers can be reproduced.
"""

import os as _os
from pathlib import Path as _Path


def _reeps_base() -> _Path:
    env = _os.environ.get("REEPS_BASE")
    if env:
        return _Path(env).expanduser().resolve()
    here = _Path(__file__).resolve()
    for cand in (here.parent, *here.parents):
        if (cand / "REEPS_Master_Database.gpkg").exists():
            return cand
    return here.parent


REEPS_BASE = _reeps_base()

import warnings
warnings.filterwarnings("ignore")

import os

import geopandas as gpd
import h3
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran_Local
from libpysal.weights import W
from shapely.geometry import Polygon

os.chdir(str(REEPS_BASE))

os.makedirs("figures", exist_ok=True)   # a fresh checkout has no figures/
SEED = 42
PERMUTATIONS = 999
QUAD = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
COLORS = {
    "High-High": "#d7191c",
    "Low-Low": "#2c7bb6",
    "High-Low": "#fdae61",
    "Low-High": "#abd9e9",
    "Not Significant": "#d9d9d9",
}


def hex_poly(cell: str) -> Polygon:
    return Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])


def build_weights(cells):
    """Row-standardised H3 ring-1 adjacency among the supplied cells."""
    idx = {c: i for i, c in enumerate(cells)}
    nb = {idx[c]: [idx[n] for n in h3.grid_ring(c, 1) if n in idx] for c in cells}
    w = W(nb, silence_warnings=True)
    w.transform = "r"
    return w


def classify(lm, alpha=0.05):
    return np.array([QUAD[q] if p < alpha else "Not Significant"
                     for q, p in zip(lm.q, lm.p_sim)])


def main() -> None:
    df = pd.read_csv("h3_analysis.csv")
    occ = df[df["species_richness"] > 0].reset_index(drop=True)
    cells = occ["h3_index"].tolist()
    y = occ["species_richness"].values.astype(float)
    print(f"Occupied cells: {len(cells)}, richness {y.min():.0f}-{y.max():.0f}")

    w = build_weights(cells)

    # Pass the seed explicitly rather than relying on the global NumPy state:
    # Moran_Local accepts one, and the global seed is not guaranteed to reach
    # esda's internal generator across versions or when n_jobs > 1.
    lm = Moran_Local(y, w, permutations=PERMUTATIONS, seed=SEED)
    labels = classify(lm)

    counts = pd.Series(labels).value_counts()
    print(f"\nLISA classification (seed {SEED}, {PERMUTATIONS} permutations):")
    print(counts.to_string())

    # How much of this survives a change of seed?
    tallies = {}
    for s in range(20):
        t = pd.Series(classify(
            Moran_Local(y, w, permutations=PERMUTATIONS, seed=s)))
        for k, v in t.value_counts().items():
            tallies.setdefault(k, []).append(v)
    print("\nacross 20 seeds:")
    for k, v in tallies.items():
        print(f"  {k:16s} min {min(v):2d}  max {max(v):2d}  "
              f"median {int(np.median(v)):2d}")

    out = pd.DataFrame({
        "h3_index": cells,
        "species_richness": y.astype(int),
        "lisa_q": lm.q,
        "lisa_p": lm.p_sim,
        "lisa_I": lm.Is,
        "lisa_label": labels,
        "seed": SEED,
    })
    out.to_csv("lisa_results.csv", index=False)
    print("\nWrote lisa_results.csv")

    gdf = gpd.GeoDataFrame(out, geometry=[hex_poly(c) for c in cells],
                           crs="EPSG:4326")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for label, color in COLORS.items():
        sub = gdf[gdf["lisa_label"] == label]
        if len(sub):
            sub.plot(ax=ax, color=color, edgecolor="black", linewidth=0.5)
    ax.set_title("(A) Local Moran's I Cluster Map\n"
                 f"(Species richness, {len(cells)} occupied cells)",
                 fontsize=11, fontweight="bold")
    handles = [mpatches.Patch(color=c, label=f"{l} (n = {int(counts.get(l, 0))})")
               for l, c in COLORS.items() if counts.get(l, 0)]
    ax.legend(handles=handles, loc="lower left", fontsize=8, title="LISA cluster")

    ax = axes[1]
    sig = gdf["lisa_p"] < 0.05
    gdf[~sig].plot(ax=ax, color="#d9d9d9", edgecolor="black", linewidth=0.5)
    if sig.any():
        gdf[sig].plot(ax=ax, color="#d7191c", edgecolor="black", linewidth=0.5)
    ax.set_title(f"(B) LISA Significance Map\n"
                 f"($p < 0.05$, {PERMUTATIONS} permutations, seed {SEED})",
                 fontsize=11, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color="#d7191c", label=f"Significant (n = {int(sig.sum())})"),
        mpatches.Patch(color="#d9d9d9",
                       label=f"Not significant (n = {int((~sig).sum())})"),
    ], loc="lower left", fontsize=8)

    for ax in axes:
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°S)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.savefig("figures/lisa_cluster_map.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("figures/lisa_cluster_map.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Wrote figures/lisa_cluster_map.pdf and .png")


if __name__ == "__main__":
    main()
