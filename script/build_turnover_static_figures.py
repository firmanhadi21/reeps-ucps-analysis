"""
build_turnover_static_figures.py
─────────────────────────────────
Generate publication-ready static maps from the REEPS Temporal Turnover Map layers:
  Fig A — Net species change (colonisations - extinctions)
  Fig B — Mean Beta_Temporal (Whittaker's temporal beta diversity)
  Fig C — Total colonisations (species gained)
  Fig D — Total local extinctions (species lost)
  Fig E — 4-panel composite

Turnover metrics are aggregated from turnover_cell_period.csv over all periods per cell.

Output folder: <project root>/static_figures/turnover/
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
OUT    = REEPS_BASE / "static_figures/turnover"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df_h      = pd.read_csv(BASE / "h3_analysis.csv")
df_turnover = pd.read_csv(BASE / "turnover_cell_period.csv")

# ── Aggregate turnover metrics per cell ─────────────────────────────────────────
turnover_agg = df_turnover.groupby("h3_index").agg(
    Colonisations=("gained", "sum"),
    Extinctions=("lost", "sum"),
    Beta_Temporal=("beta_whittaker", "mean"),
).reset_index()

# Net change = colonisations - extinctions
turnover_agg["Net_Change"] = turnover_agg["Colonisations"] - turnover_agg["Extinctions"]

# ── Build H3 polygon GeoDataFrame (all 149 cells) ────────────────────────────
all_cells = df_h["h3_index"].tolist()
geoms     = [h3_to_polygon(c) for c in all_cells]
gdf_all   = gpd.GeoDataFrame(df_h.copy(), geometry=geoms, crs="EPSG:4326")

# Merge turnover metrics
gdf_all = gdf_all.merge(
    turnover_agg,
    on="h3_index", how="left"
)

occ_mask  = gdf_all["total_records"] > 0
unocc_mask = ~occ_mask

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — Net Species Change (Diverging colormap)
# ══════════════════════════════════════════════════════════════════════════════
print("[1/5] Net species change …")

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(a)  Net Species Change (2009–2026)", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Occupied cells by net change (diverging RdYlGn, centred at 0)
occ_gdf = gdf_all[occ_mask].copy()
net_min = occ_gdf["Net_Change"].min()
net_max = occ_gdf["Net_Change"].max()
# Use symmetric limits around 0
lim = max(abs(net_min), abs(net_max))
norm_net = mcolors.TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
cmap_net = plt.cm.RdYlGn

occ_gdf["_c"] = occ_gdf["Net_Change"].apply(
    lambda v: mcolors.to_hex(cmap_net(norm_net(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf.plot(ax=ax, color=occ_gdf["_c"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Label cells with net_change != 0
for _, row in occ_gdf[occ_gdf["Net_Change"] != 0].iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    nc = row.get("Net_Change", np.nan)
    if pd.notna(nc) and nc != 0:
        sign = "+" if nc > 0 else "−"
        ax.text(cx, cy, f"{sign}{int(abs(nc))}", ha="center", va="center",
                fontsize=6, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=0.8, foreground="#4A4A4A")],
                zorder=6)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Colorbar (diverging)
cax = fig.add_axes([0.81, 0.15, 0.015, 0.6])
cb = ColorbarBase(cax, cmap=cmap_net, norm=norm_net, orientation="vertical")
cb.set_label("Net Change", fontsize=8.5)
cb.ax.tick_params(labelsize=7)

# Legend
handles = [
    mpatches.Patch(facecolor="#1B5E20", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Net gain"),
    mpatches.Patch(facecolor="#FFEB3B", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Stable"),
    mpatches.Patch(facecolor="#B71C1C", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Net loss"),
    s2_patch(),
    aoi_handle(),
    unocc_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Net species change (colonisations − extinctions) across all survey periods · "
                 "Cells with zero net change not labelled · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Turn_A_net_change")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Beta Temporal (Whittaker's temporal beta diversity)
# ══════════════════════════════════════════════════════════════════════════════
print("[2/5] Beta temporal …")

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(b)  Mean Beta_Temporal (Species Turnover)", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Occupied cells by beta temporal (Purples colormap)
occ_gdf = gdf_all[occ_mask].copy()
beta_max = occ_gdf["Beta_Temporal"].max()
norm_beta = mcolors.Normalize(vmin=0, vmax=beta_max)
cmap_beta = plt.cm.Purples

occ_gdf["_c"] = occ_gdf["Beta_Temporal"].apply(
    lambda v: mcolors.to_hex(cmap_beta(norm_beta(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf.plot(ax=ax, color=occ_gdf["_c"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Label top-5 cells (highest temporal turnover)
top5_beta = occ_gdf.nlargest(5, "Beta_Temporal")
for _, row in top5_beta.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    beta = row.get("Beta_Temporal", np.nan)
    if pd.notna(beta):
        ax.text(cx, cy, f"{beta:.2f}", ha="center", va="center",
                fontsize=6, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=0.8, foreground="#4A148C")],
                zorder=6)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Colorbar
cax = fig.add_axes([0.81, 0.15, 0.015, 0.6])
cb = ColorbarBase(cax, cmap=cmap_beta, norm=norm_beta, orientation="vertical")
cb.set_label("Beta_Temporal", fontsize=8.5)
cb.ax.tick_params(labelsize=7)

# Legend
handles = [
    s2_patch(),
    aoi_handle(),
    unocc_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Mean Whittaker's temporal beta diversity (species turnover) per cell · "
                 "Higher values = more species change between periods · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Turn_B_beta_temporal")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Total Colonisations (species gained)
# ══════════════════════════════════════════════════════════════════════════════
print("[3/5] Total colonisations …")

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(c)  Total Colonisations (2009–2026)", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Occupied cells by colonisations (Greens colormap)
occ_gdf = gdf_all[occ_mask].copy()
col_max = occ_gdf["Colonisations"].max()
norm_col = mcolors.Normalize(vmin=0, vmax=col_max)
cmap_col = plt.cm.Greens

occ_gdf["_c"] = occ_gdf["Colonisations"].apply(
    lambda v: mcolors.to_hex(cmap_col(norm_col(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf.plot(ax=ax, color=occ_gdf["_c"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Label top-5 cells
top5_col = occ_gdf.nlargest(5, "Colonisations")
for _, row in top5_col.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    col = row.get("Colonisations", np.nan)
    if pd.notna(col):
        ax.text(cx, cy, f"{int(col)}", ha="center", va="center",
                fontsize=6, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=0.8, foreground="#1B5E20")],
                zorder=6)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Colorbar
cax = fig.add_axes([0.81, 0.15, 0.015, 0.6])
cb = ColorbarBase(cax, cmap=cmap_col, norm=norm_col, orientation="vertical")
cb.set_label("Colonisations", fontsize=8.5)
cb.ax.tick_params(labelsize=7)

# Legend
handles = [
    s2_patch(),
    aoi_handle(),
    unocc_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Total species colonisations (range expansions) per cell · "
                 "Top-5 cells labelled with count · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Turn_C_colonisation")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D — Total Extinctions (species lost)
# ══════════════════════════════════════════════════════════════════════════════
print("[4/5] Total extinctions …")

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(d)  Total Local Extinctions (2009–2026)", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Occupied cells by extinctions (Reds colormap)
occ_gdf = gdf_all[occ_mask].copy()
ext_max = occ_gdf["Extinctions"].max()
norm_ext = mcolors.Normalize(vmin=0, vmax=ext_max)
cmap_ext = plt.cm.Reds

occ_gdf["_c"] = occ_gdf["Extinctions"].apply(
    lambda v: mcolors.to_hex(cmap_ext(norm_ext(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf.plot(ax=ax, color=occ_gdf["_c"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Label top-5 cells
top5_ext = occ_gdf.nlargest(5, "Extinctions")
for _, row in top5_ext.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ext = row.get("Extinctions", np.nan)
    if pd.notna(ext):
        ax.text(cx, cy, f"{int(ext)}", ha="center", va="center",
                fontsize=6, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=0.8, foreground="#5D0000")],
                zorder=6)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Colorbar
cax = fig.add_axes([0.81, 0.15, 0.015, 0.6])
cb = ColorbarBase(cax, cmap=cmap_ext, norm=norm_ext, orientation="vertical")
cb.set_label("Extinctions", fontsize=8.5)
cb.ax.tick_params(labelsize=7)

# Legend
handles = [
    s2_patch(),
    aoi_handle(),
    unocc_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Total local extinctions (range contractions) per cell · "
                 "Top-5 cells labelled with count · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Turn_D_extinction")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE E — 4-panel Composite
# ══════════════════════════════════════════════════════════════════════════════
print("[5/5] 4-panel composite …")

fig = plt.figure(figsize=(14.0, 10.5), facecolor="white")
gs = fig.add_gridspec(2, 2, hspace=0.12, wspace=0.06, left=0.05, right=0.95, top=0.95, bottom=0.05)

for ax_idx, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
    ax = fig.add_subplot(gs[row, col])
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.set_facecolor("none")
    add_graticule(ax, xlabels=ax_idx >= 2, ylabels=ax_idx % 2 == 0,
                  fontsize=6.0)
    for sp in ax.spines.values():
        sp.set_edgecolor("#888888")
        sp.set_linewidth(0.5)
    ax.grid(True, linestyle=":", linewidth=0.2, color="#BBBBBB", alpha=0.4, zorder=1)
    ax.imshow(S2_RGBA, extent=S2_EXTENT, origin="upper", aspect="equal", interpolation="bilinear", zorder=0)
    gdf_all[unocc_mask].plot(ax=ax, color=UNOCC_COLOR, edgecolor=BORDER_COLOR, linewidth=0.3, alpha=UNOCC_ALPHA, zorder=2)
    aoi.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=1.5, zorder=5, linestyle="--")

# Panel A: Net Change
ax_a = fig.get_axes()[0]
ax_a.set_title("(a)  Net Species Change", fontweight="bold", fontsize=10, loc="left")
occ_gdf_a = gdf_all[occ_mask].copy()
net_min_a = occ_gdf_a["Net_Change"].min()
net_max_a = occ_gdf_a["Net_Change"].max()
lim_a = max(abs(net_min_a), abs(net_max_a))
norm_net_a = mcolors.TwoSlopeNorm(vcenter=0, vmin=-lim_a, vmax=lim_a)
cmap_net_a = plt.cm.RdYlGn
occ_gdf_a["_c"] = occ_gdf_a["Net_Change"].apply(
    lambda v: mcolors.to_hex(cmap_net_a(norm_net_a(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf_a.plot(ax=ax_a, color=occ_gdf_a["_c"].tolist(), edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)

# Panel B: Beta Temporal
ax_b = fig.get_axes()[1]
ax_b.set_title("(b)  Beta_Temporal (Turnover)", fontweight="bold", fontsize=10, loc="left")
occ_gdf_b = gdf_all[occ_mask].copy()
beta_max_b = occ_gdf_b["Beta_Temporal"].max()
norm_beta_b = mcolors.Normalize(vmin=0, vmax=beta_max_b)
cmap_beta_b = plt.cm.Purples
occ_gdf_b["_c"] = occ_gdf_b["Beta_Temporal"].apply(
    lambda v: mcolors.to_hex(cmap_beta_b(norm_beta_b(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf_b.plot(ax=ax_b, color=occ_gdf_b["_c"].tolist(), edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)

# Panel C: Colonisations
ax_c = fig.get_axes()[2]
ax_c.set_title("(c)  Total Colonisations", fontweight="bold", fontsize=10, loc="left")
occ_gdf_c = gdf_all[occ_mask].copy()
col_max_c = occ_gdf_c["Colonisations"].max()
norm_col_c = mcolors.Normalize(vmin=0, vmax=col_max_c)
cmap_col_c = plt.cm.Greens
occ_gdf_c["_c"] = occ_gdf_c["Colonisations"].apply(
    lambda v: mcolors.to_hex(cmap_col_c(norm_col_c(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf_c.plot(ax=ax_c, color=occ_gdf_c["_c"].tolist(), edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)

# Panel D: Extinctions
ax_d = fig.get_axes()[3]
ax_d.set_title("(d)  Total Local Extinctions", fontweight="bold", fontsize=10, loc="left")
occ_gdf_d = gdf_all[occ_mask].copy()
ext_max_d = occ_gdf_d["Extinctions"].max()
norm_ext_d = mcolors.Normalize(vmin=0, vmax=ext_max_d)
cmap_ext_d = plt.cm.Reds
occ_gdf_d["_c"] = occ_gdf_d["Extinctions"].apply(
    lambda v: mcolors.to_hex(cmap_ext_d(norm_ext_d(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf_d.plot(ax=ax_d, color=occ_gdf_d["_c"].tolist(), edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)

# ── Add north arrows and scale bars to all panels ──
for _ax in [ax_a, ax_b, ax_c, ax_d]:
    add_north_arrow(_ax)
    add_scalebar(_ax)

# ── Panel A legend ──
handles_a = [
    mpatches.Patch(facecolor="#1B5E20", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Net gain"),
    mpatches.Patch(facecolor="#FFEB3B", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Stable"),
    mpatches.Patch(facecolor="#B71C1C", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Net loss"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax_a.legend(handles=handles_a, loc="upper left", fontsize=6.5, framealpha=0.90,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

# ── Panel B legend ──
handles_b = [
    mpatches.Patch(facecolor="#9C27B0", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="High turnover (Purples)"),
    mpatches.Patch(facecolor="#E8E8E8", edgecolor=BORDER_COLOR, alpha=0.6, label="Unoccupied cell"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax_b.legend(handles=handles_b, loc="upper left", fontsize=6.5, framealpha=0.90,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

# ── Panel C legend ──
handles_c = [
    mpatches.Patch(facecolor="#2E7D32", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Colonisations (Greens)"),
    mpatches.Patch(facecolor="#E8E8E8", edgecolor=BORDER_COLOR, alpha=0.6, label="Unoccupied cell"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax_c.legend(handles=handles_c, loc="upper left", fontsize=6.5, framealpha=0.90,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

# ── Panel D legend ──
handles_d = [
    mpatches.Patch(facecolor="#C62828", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Extinctions (Reds)"),
    mpatches.Patch(facecolor="#E8E8E8", edgecolor=BORDER_COLOR, alpha=0.6, label="Unoccupied cell"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax_d.legend(handles=handles_d, loc="upper left", fontsize=6.5, framealpha=0.90,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

save_fig(fig, OUT, "fig_Turn_E_composite")

print("\nAll turnover figures complete!")
