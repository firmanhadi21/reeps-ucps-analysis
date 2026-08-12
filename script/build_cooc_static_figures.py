"""
build_cooc_static_figures.py
──────────────────────────────
Generate publication-ready static maps from the REEPS Co-occurrence Map layers:
  Fig A — Number of significant co-occurring pairs per occupied cell
  Fig B — Geographic co-occurrence network (top-20 pairs)
  Fig C — Species occupancy bar chart
  Fig D — Jaccard similarity heatmap
  Fig E — 4-panel composite

Output folder: <project root>/static_figures/cooccurrence/
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

from basemap_utils import *

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = REEPS_BASE
OUT    = REEPS_BASE / "static_figures/cooccurrence"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df_h   = pd.read_csv(BASE / "h3_analysis.csv")
# presence_absence.csv: species as rows, h3 cells as columns
_pa_raw = pd.read_csv(BASE / "presence_absence.csv")
# Transpose so that h3_index is a column and species names are columns
df_pa = _pa_raw.set_index("species").T.reset_index().rename(columns={"index": "h3_index"})

df_cooc_pairs = pd.read_csv(BASE / "cooccurrence_pairs.csv")
df_cooc_spp = pd.read_csv(BASE / "cooccurrence_species.csv")

# Compute Pct_sites for species (percentage of occupied cells)
n_total_cells = len(df_h)
df_cooc_spp["Pct_sites"] = (df_cooc_spp["n_cells"] / n_total_cells * 100).round(1)

# ── Build H3 polygon GeoDataFrame (all 149 cells) ────────────────────────────
all_cells = df_h["h3_index"].tolist()
geoms     = [h3_to_polygon(c) for c in all_cells]
gdf_all   = gpd.GeoDataFrame(df_h.copy(), geometry=geoms, crs="EPSG:4326")

occ_mask  = gdf_all["total_records"] > 0
unocc_mask = ~occ_mask

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — Pair Count (number of significant co-occurring pairs per cell)
# ══════════════════════════════════════════════════════════════════════════════
print("[1/5] Co-occurrence pair count …")

# Count co-occurring pairs for each cell
pair_counts = {}
for cell in all_cells:
    row = df_pa[df_pa["h3_index"] == cell]
    if len(row) == 0:
        pair_counts[cell] = 0
        continue
    species_in_cell = [col for col in df_pa.columns if col != "h3_index" and row[col].values[0] == 1]
    # Count pairs where both species present
    count = 0
    for _, pair_row in df_cooc_pairs.iterrows():
        spp_a = pair_row["species_1"]
        spp_b = pair_row["species_2"]
        if spp_a in species_in_cell and spp_b in species_in_cell:
            count += 1
    pair_counts[cell] = count

gdf_all["pair_count"] = gdf_all["h3_index"].map(pair_counts)

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(a)  Co-occurring Pairs per Cell", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Plot occupied cells coloured by pair count (GnBu colormap)
occ_gdf = gdf_all[occ_mask].copy()
pc_max = occ_gdf["pair_count"].max()
norm_pc = mcolors.Normalize(vmin=0, vmax=max(3, pc_max))
cmap_pc = plt.cm.GnBu

occ_gdf["_c"] = occ_gdf["pair_count"].apply(
    lambda v: mcolors.to_hex(cmap_pc(norm_pc(v))))
occ_gdf.plot(ax=ax, color=occ_gdf["_c"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Label cells with count >= 3
for _, row in occ_gdf[occ_gdf["pair_count"] >= 3].iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(cx, cy, f"{int(row['pair_count'])}", ha="center", va="center",
            fontsize=6, fontweight="bold", color="white",
            path_effects=[pe.withStroke(linewidth=0.8, foreground="#1B5E20")],
            zorder=6)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Colorbar
cax = fig.add_axes([0.81, 0.15, 0.015, 0.6])
cb = ColorbarBase(cax, cmap=cmap_pc, norm=norm_pc, orientation="vertical")
cb.set_label("Pair Count", fontsize=8.5)
cb.ax.tick_params(labelsize=7)

# Legend
handles = [
    s2_patch(),
    aoi_handle(),
    unocc_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Count of significant co-occurring species pairs per occupied H3 cell · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_CoOc_A_pair_count")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Geographic Co-occurrence Network
# ══════════════════════════════════════════════════════════════════════════════
print("[2/5] Geographic co-occurrence network …")

# Top-20 co-occurring pairs
top_pairs = df_cooc_pairs.nlargest(20, "shared_cells")

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(b)  Geographic Co-occurrence Network (Top-20 Pairs)", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Plot occupied cells by richness (YlOrRd)
occ_gdf = gdf_all[occ_mask].copy()
norm_rich = mcolors.Normalize(vmin=1, vmax=occ_gdf["species_richness"].max())
cmap_rich = plt.cm.YlOrRd

occ_gdf["_c"] = occ_gdf["species_richness"].apply(
    lambda v: mcolors.to_hex(cmap_rich(norm_rich(v))))
occ_gdf.plot(ax=ax, color=occ_gdf["_c"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Draw network lines between top pairs
norm_jac = mcolors.Normalize(vmin=top_pairs["jaccard"].min(), vmax=top_pairs["jaccard"].max())
cmap_jac = plt.cm.RdYlGn

for _, pair_row in top_pairs.iterrows():
    spp_a = pair_row["species_1"]
    spp_b = pair_row["species_2"]
    # Find cells containing both species
    cells_a = df_pa[df_pa[spp_a] == 1]["h3_index"].tolist()
    cells_b = df_pa[df_pa[spp_b] == 1]["h3_index"].tolist()
    
    # Get unique cells for this pair
    cells_pair = list(set(cells_a) & set(cells_b))
    if len(cells_pair) >= 2:
        # Draw lines between pairs
        for i in range(len(cells_pair) - 1):
            c1 = gdf_all[gdf_all["h3_index"] == cells_pair[i]]
            c2 = gdf_all[gdf_all["h3_index"] == cells_pair[i + 1]]
            if len(c1) > 0 and len(c2) > 0:
                x1, y1 = c1.geometry.centroid.x.values[0], c1.geometry.centroid.y.values[0]
                x2, y2 = c2.geometry.centroid.x.values[0], c2.geometry.centroid.y.values[0]
                jac = pair_row["jaccard"]
                color = mcolors.to_hex(cmap_jac(norm_jac(jac)))
                lw = 0.5 + (pair_row["shared_cells"] / top_pairs["shared_cells"].max()) * 1.5
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.6, zorder=2)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Legend with line width scale
handles = [
    Line2D([0], [0], color="#555", lw=0.5, label="1 co-occurrence"),
    Line2D([0], [0], color="#555", lw=2.0, label=f"{int(top_pairs['shared_cells'].max())} co-occurrences"),
    s2_patch(),
    aoi_handle(),
    unocc_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Lines connecting centroids of cells where species pairs co-occur · "
                 "Line colour: Jaccard similarity (red=low, yellow=mid, green=high) · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_CoOc_B_network_geo")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Species Occupancy Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
print("[3/5] Species occupancy bar chart …")

fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")

# Sort by Pct_sites descending
df_cooc_spp_sorted = df_cooc_spp.sort_values("Pct_sites", ascending=True)

colors = [SP_COL.get(spp, "#888888") for spp in df_cooc_spp_sorted["species"]]
y_pos = np.arange(len(df_cooc_spp_sorted))

ax.barh(y_pos, df_cooc_spp_sorted["Pct_sites"], color=colors, alpha=0.8, edgecolor="#333", linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([ABBREV.get(spp, spp) for spp in df_cooc_spp_sorted["species"]], fontsize=9)
ax.set_xlabel("Site Occupancy (%)", fontsize=10, fontweight="bold")
ax.set_title("Species Site Occupancy", fontsize=11, fontweight="bold", pad=10)
ax.grid(True, axis="x", linewidth=0.3, alpha=0.5)

# Add value labels
for i, (idx, row) in enumerate(df_cooc_spp_sorted.iterrows()):
    ax.text(row["Pct_sites"] + 1, i, f"{row['Pct_sites']:.1f}%", 
            va="center", fontsize=8)

plt.tight_layout()
save_fig(fig, OUT, "fig_CoOc_C_species_occupancy")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D — Jaccard Similarity Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("[4/5] Jaccard similarity heatmap …")

# Build symmetric Jaccard matrix
species_list = sorted(set(df_cooc_pairs["species_1"].tolist()) |
                      set(df_cooc_pairs["species_2"].tolist()))
n_spp = len(species_list)
jaccard_mat = np.eye(n_spp)

for _, row in df_cooc_pairs.iterrows():
    i = species_list.index(row["species_1"])
    j = species_list.index(row["species_2"])
    jac = row["jaccard"]
    jaccard_mat[i, j] = jac
    jaccard_mat[j, i] = jac

fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")

im = ax.imshow(jaccard_mat, cmap="YlOrBr", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(np.arange(n_spp))
ax.set_yticks(np.arange(n_spp))
ax.set_xticklabels([ABBREV.get(spp, spp) for spp in species_list], rotation=45, ha="right", fontsize=8)
ax.set_yticklabels([ABBREV.get(spp, spp) for spp in species_list], fontsize=8)
ax.set_title("Jaccard Similarity Between Species Pairs", fontsize=11, fontweight="bold", pad=10)

# Annotate cells > 0.3
for i in range(n_spp):
    for j in range(n_spp):
        if jaccard_mat[i, j] > 0.3 and i != j:
            ax.text(j, i, f"{jaccard_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

plt.colorbar(im, ax=ax, label="Jaccard Index", shrink=0.8)
plt.tight_layout()
save_fig(fig, OUT, "fig_CoOc_D_jaccard_matrix")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE E — 4-panel Composite
# ══════════════════════════════════════════════════════════════════════════════
print("[5/5] 4-panel composite …")

fig = plt.figure(figsize=(14.0, 10.5), facecolor="white")

# Setup 2x2 grid with custom spacing
gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.15, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Panel A: Pair count map
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_xlim(XMIN, XMAX)
ax0.set_ylim(YMIN, YMAX)
ax0.set_aspect("equal")
ax0.set_title("(a)  Co-occurring Pairs per Cell", fontweight="bold", fontsize=10, loc="left")
ax0.set_facecolor("none")
ax0.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for sp in ax0.spines.values():
    sp.set_edgecolor("#888888")
    sp.set_linewidth(0.5)
ax0.grid(True, linestyle=":", linewidth=0.2, color="#BBBBBB", alpha=0.4, zorder=1)
ax0.imshow(S2_RGBA, extent=S2_EXTENT, origin="upper", aspect="equal", interpolation="bilinear", zorder=0)
gdf_all[unocc_mask].plot(ax=ax0, color=UNOCC_COLOR, edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=UNOCC_ALPHA, zorder=2)
occ_gdf_a = gdf_all[occ_mask].copy()
pc_max_a = occ_gdf_a["pair_count"].max()
norm_pc_a = mcolors.Normalize(vmin=0, vmax=max(3, pc_max_a))
cmap_pc_a = plt.cm.GnBu
occ_gdf_a["_c"] = occ_gdf_a["pair_count"].apply(lambda v: mcolors.to_hex(cmap_pc_a(norm_pc_a(v))))
occ_gdf_a.plot(ax=ax0, color=occ_gdf_a["_c"].tolist(), edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)
aoi.boundary.plot(ax=ax0, color=AOI_COLOR, linewidth=1.8, zorder=5, linestyle="--")
add_north_arrow(ax0)
add_scalebar(ax0)

# Panel A legend
handles_a0 = [
    mpatches.Patch(facecolor="#6BAED6", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Occupied (GnBu cmap)"),
    mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR, alpha=UNOCC_ALPHA, label="Unoccupied cell"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax0.legend(handles=handles_a0, loc="upper left", fontsize=6.5, framealpha=0.90,
           edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

# Panel B: Network map (simplified for composite)
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_xlim(XMIN, XMAX)
ax1.set_ylim(YMIN, YMAX)
ax1.set_aspect("equal")
ax1.set_title("(b)  Top-20 Co-occurrence Network", fontweight="bold", fontsize=10, loc="left")
ax1.set_facecolor("none")
ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for sp in ax1.spines.values():
    sp.set_edgecolor("#888888")
    sp.set_linewidth(0.5)
ax1.grid(True, linestyle=":", linewidth=0.2, color="#BBBBBB", alpha=0.4, zorder=1)
ax1.imshow(S2_RGBA, extent=S2_EXTENT, origin="upper", aspect="equal", interpolation="bilinear", zorder=0)
gdf_all[unocc_mask].plot(ax=ax1, color=UNOCC_COLOR, edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=UNOCC_ALPHA, zorder=2)
occ_gdf_b = gdf_all[occ_mask].copy()
norm_rich_b = mcolors.Normalize(vmin=1, vmax=occ_gdf_b["species_richness"].max())
cmap_rich_b = plt.cm.YlOrRd
occ_gdf_b["_c"] = occ_gdf_b["species_richness"].apply(lambda v: mcolors.to_hex(cmap_rich_b(norm_rich_b(v))))
occ_gdf_b.plot(ax=ax1, color=occ_gdf_b["_c"].tolist(), edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)
aoi.boundary.plot(ax=ax1, color=AOI_COLOR, linewidth=1.8, zorder=5, linestyle="--")
add_north_arrow(ax1)
add_scalebar(ax1)

# Panel B legend
handles_b1 = [
    mpatches.Patch(facecolor="#FEB24C", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Occupied (YlOrRd cmap)"),
    Line2D([0], [0], color="#555", lw=1.5, label="Co-occurrence link"),
    mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR, alpha=UNOCC_ALPHA, label="Unoccupied cell"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax1.legend(handles=handles_b1, loc="upper left", fontsize=6.5, framealpha=0.90,
           edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

# Panel C: Species occupancy chart
ax2 = fig.add_subplot(gs[1, 0])
df_cooc_spp_sorted = df_cooc_spp.sort_values("Pct_sites", ascending=True)
colors_c = [SP_COL.get(spp, "#888888") for spp in df_cooc_spp_sorted["species"]]
y_pos_c = np.arange(len(df_cooc_spp_sorted))
ax2.barh(y_pos_c, df_cooc_spp_sorted["Pct_sites"], color=colors_c, alpha=0.8, edgecolor="#333", linewidth=0.5)
ax2.set_yticks(y_pos_c)
ax2.set_yticklabels([ABBREV.get(spp, spp) for spp in df_cooc_spp_sorted["species"]], fontsize=8)
ax2.set_xlabel("Site Occupancy (%)", fontsize=9, fontweight="bold")
ax2.set_title("(c)  Species Occupancy", fontweight="bold", fontsize=10, loc="left")
ax2.grid(True, axis="x", linewidth=0.3, alpha=0.5)

# Panel D: Jaccard heatmap
ax3 = fig.add_subplot(gs[1, 1])
im = ax3.imshow(jaccard_mat, cmap="YlOrBr", aspect="auto", vmin=0, vmax=1)
ax3.set_xticks(np.arange(n_spp))
ax3.set_yticks(np.arange(n_spp))
ax3.set_xticklabels([ABBREV.get(spp, spp) for spp in species_list], rotation=45, ha="right", fontsize=7)
ax3.set_yticklabels([ABBREV.get(spp, spp) for spp in species_list], fontsize=7)
ax3.set_title("(d)  Jaccard Similarity Matrix", fontweight="bold", fontsize=10, loc="left")
for i in range(n_spp):
    for j in range(n_spp):
        if jaccard_mat[i, j] > 0.3 and i != j:
            ax3.text(j, i, f"{jaccard_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold")
plt.colorbar(im, ax=ax3, label="Jaccard", shrink=0.8)

save_fig(fig, OUT, "fig_CoOc_E_composite")

print("\nAll co-occurrence figures complete!")
