"""
build_conn_static_figures.py
────────────────────────────
Generate four publication-ready static maps from the REEPS Connectivity Map:
  Fig A — Cell Type (Core/Connected/Isolated/Unknown)
  Fig B — Isolation Distance (km to nearest occupied cell)
  Fig C — Patch Size (number of contiguous occupied cells)
  Fig D — Ecological Importance & Keystone Status
  Fig E — 4-panel composite

Each figure uses the Sentinel-2 true-colour basemap (30 Sep 2024) as background
with semi-transparent H3 hex overlays (OCC_ALPHA=0.65).

Each figure is saved as:
  • PNG at 300 DPI  (for manuscript embedding)
  • PDF (vector, for journal submission)

Output folder: <project root>/static_figures/connectivity/
"""

# ── Portable project root (auto-inserted; replaces a hardcoded absolute path) ──
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
# ──────────────────────────────────────────────────────────────────────────────


import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(REEPS_BASE))

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D
from pathlib import Path
import h3

from basemap_utils import (
    S2_RGBA, S2_EXTENT, XMIN, XMAX, YMIN, YMAX,
    BASE, BIO, STATIC_OUT, OCC_ALPHA, BORDER_COLOR, BORDER_LW,
    UNOCC_COLOR, UNOCC_ALPHA, AOI_COLOR, AOI_LW, FIG_W, FIG_H,
    h3_to_polygon, setup_ax, add_basemap, draw_hex_base,
    add_north_arrow, add_scalebar, add_subtitle, s2_patch, aoi_handle,
    unocc_handle, save_fig, aoi, std_colorbar
)

from shapely.geometry import Polygon

# ── Output directory ───────────────────────────────────────────────────────
OUT = STATIC_OUT / "connectivity"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading connectivity data …")
df_conn = pd.read_csv(BASE / "h3_connectivity_full.csv")
df_h    = pd.read_csv(BASE / "h3_analysis.csv")
df_prio = pd.read_csv(BASE / "priority_index.csv")

# ── Build H3 polygon GeoDataFrame (all 149 cells) ────────────────────────
all_cells = df_h["h3_index"].tolist()
geoms     = [h3_to_polygon(c) for c in all_cells]
gdf_all   = gpd.GeoDataFrame(df_h.copy(), geometry=geoms, crs="EPSG:4326")

# Merge connectivity data (only columns that exist in the file)
# Actual columns: h3_index, cell_type, occupied, patch_id, occ_nbrs, k2_occ_nbrs, sp_reachable_k2, eco_score
conn_cols = ["h3_index", "occupied", "patch_id", "occ_nbrs",
             "k2_occ_nbrs", "cell_type"]
conn_cols = [c for c in conn_cols if c in df_conn.columns]
gdf_all = gdf_all.merge(df_conn[conn_cols], on="h3_index", how="left",
                         suffixes=("", "_conn"))

# Derive patch_size from patch_id (count cells per patch)
if "patch_id" in gdf_all.columns:
    _ps = gdf_all.groupby("patch_id")["h3_index"].transform("count")
    gdf_all["patch_size"] = _ps.where(gdf_all["patch_id"].notna(), other=np.nan)

# Derive is_isolated (no occupied neighbours)
gdf_all["is_isolated"] = gdf_all["occ_nbrs"].fillna(0) == 0

# Load isolation distance from gap_analysis.csv (dist_to_nearest_occ_km)
df_gap = pd.read_csv(BASE / "gap_analysis.csv")
if "dist_to_nearest_occ_km" in df_gap.columns:
    gdf_all = gdf_all.merge(df_gap[["h3_index", "dist_to_nearest_occ_km"]],
                             on="h3_index", how="left")

# Merge priority index
# Actual columns: Rank, h3_index, Priority_Tier, Richness, Diversity, Connectivity,
#   Co_occurrence, Threatened, Priority_Index, lat, lon, score_richness, ...
try:
    prio_cols = ["h3_index", "Connectivity", "Priority_Tier"]
    prio_cols = [c for c in prio_cols if c in df_prio.columns]
    gdf_all = gdf_all.merge(df_prio[prio_cols], on="h3_index", how="left")
    # Map Connectivity -> ecological_importance, Priority_Tier top -> is_keystone
    if "Connectivity" in gdf_all.columns:
        gdf_all["ecological_importance"] = gdf_all["Connectivity"]
    if "Priority_Tier" in gdf_all.columns:
        gdf_all["is_keystone"] = gdf_all["Priority_Tier"].str.lower().str.contains("tier 1", na=False)
except Exception as e:
    print(f"  Warning: Could not merge priority_index: {e}")

# Ensure required columns exist with defaults
for col, default in [("ecological_importance", 0.0), ("is_keystone", False), ("cell_type", "Unknown")]:
    if col not in gdf_all.columns:
        gdf_all[col] = default

# Occupied vs unoccupied masks
occ_mask  = (gdf_all["occupied"] == True) | (gdf_all["total_records"] > 0)
unocc_mask = ~occ_mask

print(f"  All {len(gdf_all)} cells loaded, {occ_mask.sum()} occupied")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — Cell Type (Core/Connected/Isolated/Unknown)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Cell Type …")

CELLTYPE_COLORS = {
    "Core":      "#1565C0",   # dark blue — high contrast on green basemap
    "Connected": "#64B5F6",   # light blue
    "Isolated":  "#FF6F00",   # orange
    "Unknown":   "#B0BEC5",   # grey
}

def infer_cell_type(row):
    """Derive cell type from connectivity data if not in priority_index."""
    # cell_type from h3_connectivity_full.csv (via _conn suffix if duplicated)
    ct = row.get("cell_type_conn") if "cell_type_conn" in row.index else row.get("cell_type")
    if pd.notna(ct) and ct not in ("Unknown", "Empty", "Occupied"):
        return ct

    if row.get("is_isolated") == True:
        return "Isolated"
    if row.get("occ_nbrs", 0) >= 2:
        return "Core"
    if row.get("occupied") == True or row.get("total_records", 0) > 0:
        return "Connected"
    return "Unknown"

gdf_all["cell_type_final"] = gdf_all.apply(infer_cell_type, axis=1)

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(a)  Cell Type", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)

# Draw all cells: occupied get full color, unoccupied get ghost
for celltype in ["Unknown", "Isolated", "Connected", "Core"]:
    subset = gdf_all[gdf_all["cell_type_final"] == celltype]
    if len(subset) == 0:
        continue

    if celltype in CELLTYPE_COLORS:
        color = CELLTYPE_COLORS[celltype]
    else:
        color = CELLTYPE_COLORS["Unknown"]

    # Occupied cells: full opacity
    occ_subset = subset[subset.index.isin(gdf_all[occ_mask].index)]
    if len(occ_subset) > 0:
        occ_subset.plot(ax=ax, color=color, edgecolor=BORDER_COLOR,
                       linewidth=BORDER_LW + 0.1, alpha=OCC_ALPHA, zorder=3)

    # Unoccupied cells: ghost
    unocc_subset = subset[subset.index.isin(gdf_all[unocc_mask].index)]
    if len(unocc_subset) > 0:
        unocc_subset.plot(ax=ax, color=color, edgecolor=BORDER_COLOR,
                         linewidth=BORDER_LW, alpha=UNOCC_ALPHA, zorder=2)

aoi.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=AOI_LW, zorder=5, linestyle="--")

add_north_arrow(ax)
add_scalebar(ax)

# Legend
celltype_counts = gdf_all[gdf_all.index.isin(gdf_all[occ_mask].index)]["cell_type_final"].value_counts()
handles = [
    mpatches.Patch(facecolor=CELLTYPE_COLORS["Core"], edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA,
                   label=f"Core             (n = {celltype_counts.get('Core', 0)})"),
    mpatches.Patch(facecolor=CELLTYPE_COLORS["Connected"], edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA,
                   label=f"Connected    (n = {celltype_counts.get('Connected', 0)})"),
    mpatches.Patch(facecolor=CELLTYPE_COLORS["Isolated"], edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA,
                   label=f"Isolated      (n = {celltype_counts.get('Isolated', 0)})"),
    mpatches.Patch(facecolor=CELLTYPE_COLORS["Unknown"], edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA,
                   label=f"Unknown    (n = {celltype_counts.get('Unknown', 0)})"),
    aoi_handle(),
    s2_patch(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, f"Connectivity status classification of H3 cells · "
                 f"Core: ≥2 occupied neighbors; Isolated: no occupied neighbors; "
                 f"Connected: occupied but <2 neighbors · All {len(gdf_all)} cells shown · "
                 f"Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Conn_A_cell_type")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Isolation Distance
# ══════════════════════════════════════════════════════════════════════════════
print("[2/5] Isolation Distance …")

# dist_to_nearest_occ_km is already in kilometres from gap_analysis.csv
all_dists = gdf_all["dist_to_nearest_occ_km"].dropna()
all_dists = all_dists[all_dists > 0]  # exclude zeros (occupied cells themselves)

if len(all_dists) > 0:
    dist_min_km, dist_max_km = all_dists.min(), all_dists.max()
else:
    dist_min_km, dist_max_km = 0, 10

cmap_D = mpl.colormaps["OrRd"]
norm_D = mcolors.Normalize(vmin=dist_min_km, vmax=dist_max_km)

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(b)  Isolation Distance", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Draw unoccupied cells by isolation distance (already in km)
for _, row in gdf_all[unocc_mask].iterrows():
    dist_km = row.get("dist_to_nearest_occ_km", np.nan)
    if pd.isna(dist_km) or dist_km <= 0:
        continue
    color = mcolors.to_hex(cmap_D(norm_D(dist_km)))
    gpd.GeoSeries([row.geometry]).plot(ax=ax, color=color, edgecolor=BORDER_COLOR,
                                       linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Draw occupied cells as dark blue (distance = 0)
occ_gdf = gdf_all[occ_mask].copy()
occ_gdf.plot(ax=ax, color="#1565C0", edgecolor=BORDER_COLOR,
            linewidth=BORDER_LW + 0.1, alpha=OCC_ALPHA, zorder=4)

aoi.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=AOI_LW, zorder=5, linestyle="--")

# Label top 5 most isolated unoccupied cells
top_isolated = gdf_all[unocc_mask].nlargest(5, "dist_to_nearest_occ_km")
for _, row in top_isolated.iterrows():
    dist_km = row.get("dist_to_nearest_occ_km", np.nan)
    if pd.notna(dist_km) and dist_km > 0:
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        ax.text(cx, cy, f"{dist_km:.1f}",
                ha="center", va="center", fontsize=6.5, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
                zorder=6)

add_north_arrow(ax)
add_scalebar(ax)

# Colorbar
cax = fig.add_axes([0.82, 0.20, 0.025, 0.55])
cb  = ColorbarBase(cax, cmap=cmap_D, norm=norm_D, orientation="vertical")
cb.set_label("Distance to Nearest\nOccupied Cell (km)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)

# Legend
handles = [
    mpatches.Patch(facecolor="#1565C0", edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA, label="Occupied cell (distance=0)"),
    aoi_handle(),
    s2_patch(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.4)

add_subtitle(ax, f"Distance to nearest occupied H3 cell for unoccupied cells · "
                 f"Occupied cells shown in dark blue · Top 5 most-isolated cells labelled · "
                 f"Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Conn_B_isolation")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Patch Size
# ══════════════════════════════════════════════════════════════════════════════
print("[3/5] Patch Size …")

cmap_P = mpl.colormaps["YlGn"]
patch_sizes = gdf_all.loc[occ_mask, "patch_size"].dropna()
p_max = patch_sizes.max() if len(patch_sizes) > 0 else 10
norm_P = mcolors.Normalize(vmin=1, vmax=p_max)

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(c)  Patch Size", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

occ_gdf = gdf_all[occ_mask].copy()
occ_gdf["color"] = occ_gdf["patch_size"].apply(
    lambda v: mcolors.to_hex(cmap_P(norm_P(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf.plot(ax=ax, color=occ_gdf["color"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW + 0.1,
             alpha=OCC_ALPHA, zorder=3)

# Label patches with size >= 3
for _, row in occ_gdf[occ_gdf["patch_size"].fillna(0) >= 3].iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ps = row.get("patch_size", np.nan)
    if pd.notna(ps):
        ax.text(cx, cy, str(int(ps)),
                ha="center", va="center", fontsize=6.5, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
                zorder=6)

add_north_arrow(ax)
add_scalebar(ax)

# Colorbar
cax = fig.add_axes([0.82, 0.20, 0.025, 0.55])
cb  = ColorbarBase(cax, cmap=cmap_P, norm=norm_P, orientation="vertical",
                   ticks=range(1, int(p_max)+1) if p_max <= 10 else None)
cb.set_label("Patch Size\n(contiguous cells)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)

# Legend
handles = [
    unocc_handle(),
    aoi_handle(),
    s2_patch(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.4)

add_subtitle(ax, f"Number of contiguous occupied cells per patch (patch_id) · "
                 f"Patches with ≥3 cells labelled · "
                 f"Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Conn_C_patch_size")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D — Ecological Importance & Keystone Status
# ══════════════════════════════════════════════════════════════════════════════
print("[4/5] Ecological Importance …")

try:
    ecolimp_vals = gdf_all.loc[occ_mask, "ecological_importance"].dropna()
    if len(ecolimp_vals) > 0:
        ei_min, ei_max = ecolimp_vals.min(), ecolimp_vals.max()
    else:
        ei_min, ei_max = 0, 100
except Exception as e:
    print(f"  Warning: ecological_importance not available: {e}")
    ei_min, ei_max = 0, 100

norm_E = mcolors.Normalize(vmin=ei_min, vmax=ei_max)
cmap_E = mpl.colormaps["plasma"]

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(d)  Ecological Importance & Keystone Status", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

occ_gdf = gdf_all[occ_mask].copy()
try:
    occ_gdf["color"] = occ_gdf["ecological_importance"].apply(
        lambda v: mcolors.to_hex(cmap_E(norm_E(v))) if pd.notna(v) else "#CCCCCC")
    occ_gdf.plot(ax=ax, color=occ_gdf["color"].tolist(),
                 edgecolor=BORDER_COLOR, linewidth=BORDER_LW + 0.1,
                 alpha=OCC_ALPHA, zorder=3)
except Exception as e:
    print(f"  Warning: Could not color by ecological importance: {e}")
    occ_gdf.plot(ax=ax, color="#A9A9A9", edgecolor=BORDER_COLOR,
                 linewidth=BORDER_LW + 0.1, alpha=OCC_ALPHA, zorder=3)

# Mark keystone cells with thick red border and label
keystone_cells = occ_gdf[occ_gdf["is_keystone"] == True]
for _, row in keystone_cells.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(cx, cy, "K",
            ha="center", va="center", fontsize=7, fontweight="bold", color="white",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="red")],
            zorder=6)

# Draw red border around keystone cells (on top)
if len(keystone_cells) > 0:
    keystone_cells.plot(ax=ax, facecolor="none", edgecolor="red",
                       linewidth=1.8, zorder=7)

aoi.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=AOI_LW, zorder=5, linestyle="--")

add_north_arrow(ax)
add_scalebar(ax)

# Colorbar
cax = fig.add_axes([0.82, 0.20, 0.025, 0.55])
try:
    cb  = ColorbarBase(cax, cmap=cmap_E, norm=norm_E, orientation="vertical")
    cb.set_label("Ecological\nImportance", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
except Exception as e:
    print(f"  Warning: Could not create colorbar: {e}")

# Legend
keystone_count = len(keystone_cells)
handles = [
    mpatches.Patch(facecolor="none", edgecolor="red", linewidth=1.8,
                   label=f"Keystone cell (n = {keystone_count})"),
    unocc_handle(),
    aoi_handle(),
    s2_patch(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.4)

add_subtitle(ax, f"Ecological importance score for occupied cells (Plasma colormap) · "
                 f"Keystone cells highlighted with red border and marked 'K' · "
                 f"Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Conn_D_keystone")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE E — 4-panel composite (manuscript-ready)
# ══════════════════════════════════════════════════════════════════════════════
print("[5/5] 4-panel composite …")

fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.5), facecolor="white",
                          gridspec_kw={"hspace": 0.12, "wspace": 0.06})
fig.patch.set_facecolor("white")

PANEL_TITLES = [
    "(a)  Cell Type",
    "(b)  Isolation Distance",
    "(c)  Patch Size",
    "(d)  Ecological Importance",
]

for ax in axes.flat:
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.set_facecolor("none")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#AAAAAA")
        sp.set_linewidth(0.5)
    ax.grid(True, linestyle=":", linewidth=0.2, color="#BBBBBB", alpha=0.4, zorder=1)
    # Basemap
    ax.imshow(S2_RGBA, extent=S2_EXTENT, origin="upper", aspect="equal",
              interpolation="bilinear", zorder=0)
    # Unoccupied cells
    gdf_all[unocc_mask].plot(ax=ax, color=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                             linewidth=0.3, alpha=UNOCC_ALPHA, zorder=2)
    # AOI
    aoi.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=1.5, zorder=5, linestyle="--")

# Panel A — Cell Type
ax0 = axes[0, 0]
celltype_counts = gdf_all[gdf_all.index.isin(gdf_all[occ_mask].index)]["cell_type_final"].value_counts()
for celltype in ["Unknown", "Isolated", "Connected", "Core"]:
    subset = gdf_all[gdf_all["cell_type_final"] == celltype]
    if len(subset) == 0:
        continue
    color = CELLTYPE_COLORS.get(celltype, CELLTYPE_COLORS["Unknown"])
    occ_subset = subset[subset.index.isin(gdf_all[occ_mask].index)]
    if len(occ_subset) > 0:
        occ_subset.plot(ax=ax0, color=color, edgecolor="white",
                       linewidth=1.2, alpha=0.85, zorder=3)

# Legend for Panel A
handles_a = [
    mpatches.Patch(facecolor=CELLTYPE_COLORS["Core"], edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA,
                   label=f"Core (n={celltype_counts.get('Core', 0)})"),
    mpatches.Patch(facecolor=CELLTYPE_COLORS["Connected"], edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA,
                   label=f"Connected (n={celltype_counts.get('Connected', 0)})"),
    mpatches.Patch(facecolor=CELLTYPE_COLORS["Isolated"], edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA,
                   label=f"Isolated (n={celltype_counts.get('Isolated', 0)})"),
    mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                   alpha=UNOCC_ALPHA,
                   label=f"Unsurveyed (n={unocc_mask.sum()})"),
]
ax0.legend(handles=handles_a, loc="upper left", fontsize=6.5, framealpha=0.90,
           edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25,
           borderpad=0.4, fancybox=True)

# Panel B — Isolation Distance
ax1 = axes[0, 1]
for _, row in gdf_all[unocc_mask].iterrows():
    dist_km = row.get("dist_to_nearest_occ_km", np.nan)
    if pd.isna(dist_km) or dist_km <= 0:
        continue
    color = mcolors.to_hex(cmap_D(norm_D(dist_km)))
    gpd.GeoSeries([row.geometry]).plot(ax=ax1, color=color, edgecolor=BORDER_COLOR,
                                       linewidth=0.3, alpha=OCC_ALPHA, zorder=3)
occ_gdf_b = gdf_all[occ_mask].copy()
occ_gdf_b.plot(ax=ax1, color="#1565C0", edgecolor=BORDER_COLOR,
              linewidth=0.3, alpha=OCC_ALPHA, zorder=4)

cax_b = fig.add_axes([0.92, 0.565, 0.012, 0.36])
cb_b = ColorbarBase(cax_b, cmap=cmap_D, norm=norm_D, orientation="vertical")
cb_b.set_label("Distance to Nearest\nOccupied Cell (km)", fontsize=7.5)
cb_b.ax.tick_params(labelsize=7)

# Legend for Panel B
handles_b = [
    mpatches.Patch(facecolor="#1565C0", edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA, label=f"Occupied (n={occ_mask.sum()})"),
    mpatches.Patch(facecolor=mcolors.to_hex(cmap_D(0.2)), edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA, label="Near (low distance)"),
    mpatches.Patch(facecolor=mcolors.to_hex(cmap_D(0.8)), edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA, label="Far (high distance)"),
]
ax1.legend(handles=handles_b, loc="upper left", fontsize=6.5, framealpha=0.90,
           edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25,
           borderpad=0.4, fancybox=True)

# Panel C — Patch Size
ax2 = axes[1, 0]
occ_gdf_c = gdf_all[occ_mask].copy()
occ_gdf_c["_c"] = occ_gdf_c["patch_size"].apply(
    lambda v: mcolors.to_hex(cmap_P(norm_P(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf_c.plot(ax=ax2, color=occ_gdf_c["_c"].tolist(),
              edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)

cax_c = fig.add_axes([0.48, 0.08, 0.012, 0.36])
cb_c = ColorbarBase(cax_c, cmap=cmap_P, norm=norm_P, orientation="vertical")
cb_c.set_label("Patch Size\n(contiguous cells)", fontsize=7.5)
cb_c.ax.tick_params(labelsize=7)

# Legend for Panel C
handles_c = [
    mpatches.Patch(facecolor=mcolors.to_hex(cmap_P(1.0)), edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA, label=f"Main patch (n={int(p_max)})"),
    mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                   alpha=UNOCC_ALPHA, label="Unsurveyed"),
]
ax2.legend(handles=handles_c, loc="upper left", fontsize=6.5, framealpha=0.90,
           edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25,
           borderpad=0.4, fancybox=True)

# Panel D — Ecological Importance
ax3 = axes[1, 1]
occ_gdf_d = gdf_all[occ_mask].copy()
try:
    occ_gdf_d["_c"] = occ_gdf_d["ecological_importance"].apply(
        lambda v: mcolors.to_hex(cmap_E(norm_E(v))) if pd.notna(v) else "#CCCCCC")
    occ_gdf_d.plot(ax=ax3, color=occ_gdf_d["_c"].tolist(),
                  edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)
except Exception as e:
    print(f"  Panel D: {e}")
    occ_gdf_d.plot(ax=ax3, color="#A9A9A9", edgecolor=BORDER_COLOR,
                  linewidth=0.3, alpha=OCC_ALPHA, zorder=3)

# Mark keystone cells
ks_cells = gdf_all[gdf_all["is_keystone"] == True]
if len(ks_cells) > 0:
    ks_cells.plot(ax=ax3, facecolor="none", edgecolor="red",
                 linewidth=1.5, zorder=7)
for _, row in ks_cells.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax3.text(cx, cy, "K", ha="center", va="center", fontsize=6,
            fontweight="bold", color="white",
            path_effects=[pe.withStroke(linewidth=1, foreground="red")], zorder=6)

cax_d = fig.add_axes([0.92, 0.08, 0.012, 0.36])
try:
    cb_d = ColorbarBase(cax_d, cmap=cmap_E, norm=norm_E, orientation="vertical")
    cb_d.set_label("Ecological\nImportance", fontsize=7.5)
    cb_d.ax.tick_params(labelsize=7)
except Exception as e:
    print(f"  Colorbar D: {e}")

# Legend for Panel D
keystone_count = len(ks_cells)
handles_d = [
    mpatches.Patch(facecolor=mcolors.to_hex(cmap_E(0.8)), edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA, label="High importance"),
    mpatches.Patch(facecolor=mcolors.to_hex(cmap_E(0.2)), edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA, label="Low importance"),
    mpatches.Patch(facecolor="none", edgecolor="red", linewidth=1.5,
                   label=f"Keystone (n={keystone_count})"),
]
ax3.legend(handles=handles_d, loc="upper left", fontsize=6.5, framealpha=0.90,
           edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25,
           borderpad=0.4, fancybox=True)

# Titles, scalebars, north arrows on each panel
for i, (ax, title) in enumerate(zip(axes.flat, PANEL_TITLES)):
    ax.set_title(title, loc="left", pad=4, fontsize=10, fontweight="bold")
    # Add scalebar to all panels
    lat_mid = (YMIN+YMAX)/2
    deg2km  = 1/(111.0*np.cos(np.radians(abs(lat_mid))))
    blen    = 2*deg2km
    bx = XMIN + (XMAX-XMIN)*0.85
    by = YMIN + (YMAX-YMIN)*0.05
    ax.fill_between([bx - blen*0.05, bx + blen*1.05],
                    by - (YMAX-YMIN)*0.025, by + (YMAX-YMIN)*0.05,
                    color="white", alpha=0.7, zorder=8, linewidth=0)
    ax.plot([bx, bx+blen], [by, by], "k-", lw=2, solid_capstyle="butt", zorder=9)
    ax.text(bx+blen/2, by+(YMAX-YMIN)*0.015, "2 km",
            ha="center", fontsize=7, zorder=9)
    # Add north arrow to all panels
    nx = XMAX-(XMAX-XMIN)*0.07
    ny = YMAX-(YMAX-YMIN)*0.10
    ax.annotate("", xy=(nx, ny), xytext=(nx, ny-(YMAX-YMIN)*0.07),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    ax.text(nx, ny+(YMAX-YMIN)*0.01, "N",
            ha="center", va="bottom", fontsize=8, fontweight="bold")

fig.text(0.5, 0.005,
         "H3 Res-8 connectivity analysis for REEPS species distribution, "
         "UCPS Landscape, West Java · 149 hexagonal cells · "
         "Basemap: Sentinel-2 true colour (30 Sep 2024)",
         ha="center", fontsize=8, color="#444444", style="italic")

save_fig(fig, OUT, "fig_Conn_E_composite")

print(f"\n✓ All connectivity static figures saved to {OUT}")
print(f"  Files: {', '.join(sorted([p.name for p in OUT.glob('fig_Conn_*.png')]))}")
