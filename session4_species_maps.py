#!/usr/bin/env python3
"""Figure 7 — distribution of the three Critically Endangered species, shaded by the
survey period in which each cell yielded its first record of that species.

Rewritten in revision. Two changes of substance:

  * Data source. The figure previously read the GeoPackage layers, which were not
    refreshed after the coordinate correction, so its panel titles reported 73
    records / 15 cells for Javan Slow Loris and 49 / 18 for Sunda Pangolin against
    the 96 / 18 and 52 / 19 that Table 1 gives. It now reads the same analysis
    outputs as every other figure, and the counts are therefore Table 1's counts.
  * Framing. The title said "Temporal Expansion", which the shading cannot support:
    roughly two thirds of these first detections coincide with the first survey to
    detect anything at all in that cell (Reviewer 2, comment 4). The figure now
    states what it shows — when a cell was first surveyed productively for the
    species — and says so in the subtitle.
"""

import os as _os
from pathlib import Path as _Path


def _reeps_base() -> _Path:
    """Locate the project root: $REEPS_BASE, else walk up to the marker file."""
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
import pandas as pd
from matplotlib.ticker import FuncFormatter
from shapely.geometry import Polygon

os.chdir(str(REEPS_BASE))

os.makedirs("figures", exist_ok=True)   # a fresh checkout has no figures/
CR_SPECIES = {
    "Panthera pardus melas": "Javan Leopard",
    "Nycticebus javanicus": "Javan Slow Loris",
    "Manis javanica": "Sunda Pangolin",
}

PERIOD_COLORS = {
    "Pre-2017": "#fee5d9",
    "2017-2018": "#fcae91",
    "2020": "#fb6a4a",
    "2022-2026": "#cb181d",
}


def assign_period(yr):
    if yr <= 2014:
        return "Pre-2017"
    if yr <= 2018:
        return "2017-2018"
    if yr <= 2020:
        return "2020"
    return "2022-2026"


def dms(value, axis):
    hemi = ("N" if value >= 0 else "S") if axis == "lat" else ("E" if value >= 0 else "W")
    total_sec = round(abs(value) * 3600)
    d, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    return f"{d}°{m:02d}'{s:02d}\"{hemi}" if s else f"{d}°{m:02d}'{hemi}"


def hex_poly(cell: str) -> Polygon:
    return Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])


# ── Data: the corrected analysis outputs, not the GeoPackage ──────────────────
occ = pd.read_csv("reeps_h3.csv")
h3a = pd.read_csv("h3_analysis.csv")
occ["Year"] = pd.to_numeric(occ["Year"], errors="coerce")

gdf_all = gpd.GeoDataFrame(
    h3a[["h3_index"]].copy(),
    geometry=[hex_poly(c) for c in h3a["h3_index"]],
    crs="EPSG:4326",
)
print(f"grid: {len(gdf_all)} cells; occurrences: {len(occ)}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

for idx, (sp, common) in enumerate(CR_SPECIES.items()):
    ax = axes[idx]
    sp_data = occ[occ["Species"] == sp]

    first = (sp_data.groupby("h3_index")
             .agg(first_year=("Year", "min"), records=("Species", "count"))
             .reset_index())
    gdf_sp = gdf_all.merge(first, on="h3_index", how="left")

    gdf_sp[gdf_sp["first_year"].isna()].plot(
        color="#f0f0f0", edgecolor="#cccccc", linewidth=0.3, ax=ax)

    occupied = gdf_sp[gdf_sp["first_year"].notna()].copy()
    occupied["period"] = occupied["first_year"].apply(assign_period)
    for period, color in PERIOD_COLORS.items():
        mask = occupied["period"] == period
        if mask.any():
            occupied[mask].plot(color=color, edgecolor="black",
                                linewidth=0.5, ax=ax)

    n_cells, n_records = len(occupied), len(sp_data)
    print(f"  {common:18s} {n_records:3d} records, {n_cells:2d} cells")
    ax.set_title(f"{common}\n({n_records} records, {n_cells} cells)",
                 fontsize=11, fontweight="bold")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: dms(v, "lon")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: dms(v, "lat")))
    ax.set_xlabel("Longitude", fontsize=8)
    if idx == 0:
        ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", linewidth=0.4, color="#BBBBBB", alpha=0.6)

    if idx == 0:
        patches = [mpatches.Patch(color=c, label=l)
                   for l, c in PERIOD_COLORS.items()]
        patches.append(mpatches.Patch(color="#f0f0f0", edgecolor="#cccccc",
                                      label="Unsurveyed / not detected"))
        ax.legend(handles=patches, loc="lower left", fontsize=7,
                  title="Period of first record")

fig.suptitle("Period of first record of each Critically Endangered species, "
             "by H3 cell", fontsize=13, fontweight="bold")
fig.text(0.5, 0.925,
         "Shading records when a cell was first surveyed productively for the "
         "species as much as when the species arrived; it is not evidence of "
         "range expansion.",
         ha="center", fontsize=9, style="italic", color="#444444")

plt.tight_layout(rect=(0, 0, 1, 0.94))
plt.savefig("figures/cr_species_distributions.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figures/cr_species_distributions.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figure saved to figures/cr_species_distributions.pdf")
