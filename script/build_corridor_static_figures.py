"""
build_corridor_static_figures.py
─────────────────────────────────
Generate publication-ready static maps from the REEPS Corridor Map layers:
  Fig A — Stepping-stone betweenness score
  Fig B — Corridor tier categorical
  Fig C — Gap priority score
  Fig D — K2 richness reachable through stepping stones
  Fig E — 4-panel composite

Output folder: <project root>/static_figures/corridor/
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
OUT    = REEPS_BASE / "static_figures/corridor"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df_h   = pd.read_csv(BASE / "h3_analysis.csv")
df_ss  = pd.read_csv(BASE / "stepping_stones.csv")
df_gap = pd.read_csv(BASE / "gap_analysis.csv")

# ── Build H3 polygon GeoDataFrame (all 149 cells) ────────────────────────────
all_cells = df_h["h3_index"].tolist()
geoms     = [h3_to_polygon(c) for c in all_cells]
gdf_all   = gpd.GeoDataFrame(df_h.copy(), geometry=geoms, crs="EPSG:4326")

# Merge stepping stones and gap analysis
gdf_all = gdf_all.merge(
    df_ss[["h3_index", "betweenness", "k2_richness_reachable", "stepping_stone_score", "corridor_tier"]],
    on="h3_index", how="left"
)
gdf_all = gdf_all.merge(
    df_gap[["h3_index", "gap_priority_score"]],
    on="h3_index", how="left"
)

occ_mask  = gdf_all["total_records"] > 0
unocc_mask = ~occ_mask

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — Betweenness Score (Oranges cmap for unoccupied)
# ══════════════════════════════════════════════════════════════════════════════
print("[1/5] Betweenness score …")

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(a)  Stepping-Stone Betweenness Score", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Unoccupied gap cells colored by betweenness
unocc_gdf = gdf_all[unocc_mask].copy()
bet_max = unocc_gdf["betweenness"].max()
norm_bet = mcolors.Normalize(vmin=0, vmax=bet_max)
cmap_bet = plt.cm.Oranges

unocc_gdf["_c"] = unocc_gdf["betweenness"].apply(
    lambda v: mcolors.to_hex(cmap_bet(norm_bet(v))) if pd.notna(v) else UNOCC_COLOR)
unocc_gdf.plot(ax=ax, color=unocc_gdf["_c"].tolist(),
               edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Occupied cells: fixed dark blue
occ_gdf = gdf_all[occ_mask].copy()
occ_gdf.plot(ax=ax, color="#1A237E", edgecolor=BORDER_COLOR,
             linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Label top-5 stepping stone cells
top5_ss = unocc_gdf.nlargest(5, "stepping_stone_score")
for _, row in top5_ss.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    score = row.get("stepping_stone_score", np.nan)
    if pd.notna(score):
        ax.text(cx, cy, f"{score:.0f}", ha="center", va="center",
                fontsize=6, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=0.8, foreground="#5D4037")],
                zorder=6)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Colorbar
cax = fig.add_axes([0.81, 0.15, 0.015, 0.6])
cb = ColorbarBase(cax, cmap=cmap_bet, norm=norm_bet, orientation="vertical")
cb.set_label("betweenness", fontsize=8.5)
cb.ax.tick_params(labelsize=7)

# Legend
handles = [
    mpatches.Patch(facecolor="#1A237E", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Occupied cell"),
    mpatches.Patch(facecolor="#FFF3E0", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Gap cell (unoccupied)"),
    s2_patch(),
    aoi_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Betweenness centrality for unoccupied gap cells · "
                 "Top-5 stepping stones labelled with score · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Corr_A_betweenness")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Corridor Tier Categorical
# ══════════════════════════════════════════════════════════════════════════════
print("[2/5] Corridor tier categorical …")

TIER_COLORS = {
    "Priority 1 — Critical Corridor": "#B71C1C",
    "Priority 2 — High-Value Corridor": "#E65100",
    "Priority 3 — Secondary Corridor": "#F9A825",
    "None": "#ECEFF1",
}

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(b)  Corridor Tier Category", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Unoccupied cells by tier
unocc_gdf = gdf_all[unocc_mask].copy()
for tier, color in TIER_COLORS.items():
    subset = unocc_gdf[unocc_gdf["corridor_tier"] == tier]
    if len(subset) == 0:
        continue
    subset.plot(ax=ax, color=color, edgecolor=BORDER_COLOR,
                linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Occupied cells: fixed green
occ_gdf = gdf_all[occ_mask].copy()
occ_gdf.plot(ax=ax, color="#2E7D32", edgecolor=BORDER_COLOR,
             linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Legend with counts
tier_counts = unocc_gdf["corridor_tier"].value_counts()
handles = [
    mpatches.Patch(facecolor="#2E7D32", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"Occupied cell (n = {len(occ_gdf)})"),
    mpatches.Patch(facecolor="#B71C1C", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"Tier 1 — Critical (n = {tier_counts.get('Priority 1 — Critical Corridor', 0)})"),
    mpatches.Patch(facecolor="#E65100", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"Tier 2 — High-value (n = {tier_counts.get('Priority 2 — High-Value Corridor', 0)})"),
    mpatches.Patch(facecolor="#F9A825", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"Tier 3 — Secondary (n = {tier_counts.get('Priority 3 — Secondary Corridor', 0)})"),
    s2_patch(),
    aoi_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Corridor priority tier for unoccupied cells · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Corr_B_corridor_tier")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Gap Priority Score
# ══════════════════════════════════════════════════════════════════════════════
print("[3/5] Gap priority score …")

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(c)  Gap Priority Score (All Unoccupied Cells)", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# All unoccupied cells by gap priority
unocc_gdf = gdf_all[unocc_mask].copy()
gap_max = unocc_gdf["gap_priority_score"].max()
norm_gap = mcolors.Normalize(vmin=0, vmax=gap_max)
cmap_gap = plt.cm.YlOrRd

unocc_gdf["_c"] = unocc_gdf["gap_priority_score"].apply(
    lambda v: mcolors.to_hex(cmap_gap(norm_gap(v))) if pd.notna(v) else UNOCC_COLOR)
unocc_gdf.plot(ax=ax, color=unocc_gdf["_c"].tolist(),
               edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Occupied cells: light gray background
occ_gdf = gdf_all[occ_mask].copy()
occ_gdf.plot(ax=ax, color="#E8E8E8", edgecolor=BORDER_COLOR,
             linewidth=BORDER_LW, alpha=0.3, zorder=2)

# Label top-5 gap priority cells
top5_gap = unocc_gdf.nlargest(5, "gap_priority_score")
for _, row in top5_gap.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    score = row.get("gap_priority_score", np.nan)
    if pd.notna(score):
        ax.text(cx, cy, f"{score:.0f}", ha="center", va="center",
                fontsize=6, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=0.8, foreground="#5D4037")],
                zorder=6)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Colorbar
cax = fig.add_axes([0.81, 0.15, 0.015, 0.6])
cb = ColorbarBase(cax, cmap=cmap_gap, norm=norm_gap, orientation="vertical")
cb.set_label("Priority Score", fontsize=8.5)
cb.ax.tick_params(labelsize=7)

# Legend
handles = [
    s2_patch(),
    aoi_handle(),
    unocc_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Gap priority score for survey planning · Top-5 priority cells labelled · "
                 "Higher score = higher survey priority · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Corr_C_gap_priority")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D — K2 Richness Reachable
# ══════════════════════════════════════════════════════════════════════════════
print("[4/5] K2 richness reachable …")

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(d)  K2 Species Richness Reachable", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Unoccupied cells by K2 richness
unocc_gdf = gdf_all[unocc_mask].copy()
k2_max = unocc_gdf["k2_richness_reachable"].max()
norm_k2 = mcolors.Normalize(vmin=0, vmax=k2_max)
cmap_k2 = plt.cm.Blues

unocc_gdf["_c"] = unocc_gdf["k2_richness_reachable"].apply(
    lambda v: mcolors.to_hex(cmap_k2(norm_k2(v))) if pd.notna(v) else UNOCC_COLOR)
unocc_gdf.plot(ax=ax, color=unocc_gdf["_c"].tolist(),
               edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Occupied cells: fixed green as reference
occ_gdf = gdf_all[occ_mask].copy()
occ_gdf.plot(ax=ax, color="#2E7D32", edgecolor=BORDER_COLOR,
             linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Colorbar
cax = fig.add_axes([0.81, 0.15, 0.015, 0.6])
cb = ColorbarBase(cax, cmap=cmap_k2, norm=norm_k2, orientation="vertical")
cb.set_label("Species Count", fontsize=8.5)
cb.ax.tick_params(labelsize=7)

# Legend
handles = [
    mpatches.Patch(facecolor="#2E7D32", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Occupied cell"),
    s2_patch(),
    aoi_handle(),
    unocc_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Species richness reachable via K2 stepping stone neighbors · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Corr_D_reachable_richness")

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
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#888888")
        sp.set_linewidth(0.5)
    ax.grid(True, linestyle=":", linewidth=0.2, color="#BBBBBB", alpha=0.4, zorder=1)
    ax.imshow(S2_RGBA, extent=S2_EXTENT, origin="upper", aspect="equal", interpolation="bilinear", zorder=0)
    gdf_all[unocc_mask].plot(ax=ax, color=UNOCC_COLOR, edgecolor=BORDER_COLOR, linewidth=0.3, alpha=UNOCC_ALPHA, zorder=2)
    aoi.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=1.5, zorder=5, linestyle="--")

# Panel A: Betweenness
ax_a = fig.get_axes()[0]
ax_a.set_title("(a)  Stepping-Stone Betweenness", fontweight="bold", fontsize=10, loc="left")
unocc_gdf_a = gdf_all[unocc_mask].copy()
bet_max_a = unocc_gdf_a["betweenness"].max()
norm_bet_a = mcolors.Normalize(vmin=0, vmax=bet_max_a)
cmap_bet_a = plt.cm.Oranges
unocc_gdf_a["_c"] = unocc_gdf_a["betweenness"].apply(
    lambda v: mcolors.to_hex(cmap_bet_a(norm_bet_a(v))) if pd.notna(v) else UNOCC_COLOR)
unocc_gdf_a.plot(ax=ax_a, color=unocc_gdf_a["_c"].tolist(), edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)
occ_gdf.plot(ax=ax_a, color="#1A237E", edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)

# Panel B: Corridor Tier
ax_b = fig.get_axes()[1]
ax_b.set_title("(b)  Corridor Tier", fontweight="bold", fontsize=10, loc="left")
unocc_gdf_b = gdf_all[unocc_mask].copy()
for tier, color in TIER_COLORS.items():
    subset = unocc_gdf_b[unocc_gdf_b["corridor_tier"] == tier]
    if len(subset) == 0:
        continue
    subset.plot(ax=ax_b, color=color, edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)
occ_gdf.plot(ax=ax_b, color="#2E7D32", edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)

# Panel C: Gap Priority
ax_c = fig.get_axes()[2]
ax_c.set_title("(c)  Gap Priority Score", fontweight="bold", fontsize=10, loc="left")
unocc_gdf_c = gdf_all[unocc_mask].copy()
gap_max_c = unocc_gdf_c["gap_priority_score"].max()
norm_gap_c = mcolors.Normalize(vmin=0, vmax=gap_max_c)
cmap_gap_c = plt.cm.YlOrRd
unocc_gdf_c["_c"] = unocc_gdf_c["gap_priority_score"].apply(
    lambda v: mcolors.to_hex(cmap_gap_c(norm_gap_c(v))) if pd.notna(v) else UNOCC_COLOR)
unocc_gdf_c.plot(ax=ax_c, color=unocc_gdf_c["_c"].tolist(), edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)
occ_gdf.plot(ax=ax_c, color="#E8E8E8", edgecolor=BORDER_COLOR, linewidth=0.3, alpha=0.3, zorder=2)

# Panel D: K2 Richness
ax_d = fig.get_axes()[3]
ax_d.set_title("(d)  K2 Species Richness Reachable", fontweight="bold", fontsize=10, loc="left")
unocc_gdf_d = gdf_all[unocc_mask].copy()
k2_max_d = unocc_gdf_d["k2_richness_reachable"].max()
norm_k2_d = mcolors.Normalize(vmin=0, vmax=k2_max_d)
cmap_k2_d = plt.cm.Blues
unocc_gdf_d["_c"] = unocc_gdf_d["k2_richness_reachable"].apply(
    lambda v: mcolors.to_hex(cmap_k2_d(norm_k2_d(v))) if pd.notna(v) else UNOCC_COLOR)
unocc_gdf_d.plot(ax=ax_d, color=unocc_gdf_d["_c"].tolist(), edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)
occ_gdf.plot(ax=ax_d, color="#2E7D32", edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)

# ── Add north arrows and scale bars to all panels ──
for _ax in [ax_a, ax_b, ax_c, ax_d]:
    add_north_arrow(_ax)
    add_scalebar(_ax)

# ── Panel A legend ──
handles_a = [
    mpatches.Patch(facecolor="#1A237E", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Occupied cell"),
    mpatches.Patch(facecolor="#FFF3E0", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Gap cell (Oranges cmap)"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax_a.legend(handles=handles_a, loc="upper left", fontsize=6.5, framealpha=0.90,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

# ── Panel B legend ──
handles_b = [
    mpatches.Patch(facecolor="#2E7D32", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Occupied cell"),
    mpatches.Patch(facecolor="#B71C1C", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Tier 1 — Critical"),
    mpatches.Patch(facecolor="#E65100", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Tier 2 — High-value"),
    mpatches.Patch(facecolor="#F9A825", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Tier 3 — Secondary"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax_b.legend(handles=handles_b, loc="upper left", fontsize=6.5, framealpha=0.90,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

# ── Panel C legend ──
handles_c = [
    mpatches.Patch(facecolor="#E8E8E8", edgecolor=BORDER_COLOR, alpha=0.6, label="Occupied cell (grey)"),
    mpatches.Patch(facecolor="#FEB24C", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Gap cell (YlOrRd cmap)"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax_c.legend(handles=handles_c, loc="upper left", fontsize=6.5, framealpha=0.90,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

# ── Panel D legend ──
handles_d = [
    mpatches.Patch(facecolor="#2E7D32", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Occupied cell"),
    mpatches.Patch(facecolor="#6BAED6", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Gap cell (Blues cmap)"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax_d.legend(handles=handles_d, loc="upper left", fontsize=6.5, framealpha=0.90,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

save_fig(fig, OUT, "fig_Corr_E_composite")

print("\nAll corridor figures complete!")
