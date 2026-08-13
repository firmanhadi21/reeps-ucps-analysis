"""
build_priority_static_figures.py
─────────────────────────────────
Generate publication-ready static maps from the REEPS Priority Map layers:
  Fig A — Conservation Priority Index (0–1)
  Fig B — Priority Tier categorical
  Fig C — Survey effort tier
  Fig D — Flagship species presence
  Fig E — 4-panel composite

Output folder: <project root>/static_figures/priority/
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
OUT    = REEPS_BASE / "static_figures/priority"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df_h   = pd.read_csv(BASE / "h3_analysis.csv")
df_pri = pd.read_csv(BASE / "priority_index.csv")
df_sur = pd.read_csv(BASE / "survey_coverage.csv")
df_gap = pd.read_csv(BASE / "gap_analysis.csv")

# ── Build H3 polygon GeoDataFrame (all 149 cells) ────────────────────────────
all_cells = df_h["h3_index"].tolist()
geoms     = [h3_to_polygon(c) for c in all_cells]
gdf_all   = gpd.GeoDataFrame(df_h.copy(), geometry=geoms, crs="EPSG:4326")

# Merge priority data — only columns that exist in the CSV
_pri_base = ["h3_index", "Priority_Index", "Priority_Tier", "Rank",
             "score_richness", "score_diversity", "score_connectivity",
             "score_cooccurrence", "score_threatened"]
_flagship  = ["has_pant_pard", "has_hylo_molo", "has_nyct_java", "has_mani_java", "has_nisa_bart"]
_pri_cols  = [c for c in _pri_base + _flagship if c in df_pri.columns]
gdf_all = gdf_all.merge(df_pri[_pri_cols], on="h3_index", how="left")
# Ensure flagship columns exist (derive from reeps_h3 if missing from priority CSV)
_fs_map = {"has_pant_pard": "Panthera pardus melas",
           "has_hylo_molo": "Hylobates moloch",
           "has_nyct_java": "Nycticebus javanicus",
           "has_mani_java": "Manis javanica",
           "has_nisa_bart": "Nisaetus bartelsi"}
df_r_fs = pd.read_csv(BASE / "reeps_h3.csv")
for col, spp in _fs_map.items():
    if col not in gdf_all.columns:
        cells_with_spp = set(df_r_fs[df_r_fs["Species"] == spp]["h3_index"].unique())
        gdf_all[col] = gdf_all["h3_index"].apply(lambda x: 1.0 if x in cells_with_spp else np.nan)
gdf_all = gdf_all.merge(
    df_sur[["H3_Index", "Effort_Tier"]],
    left_on="h3_index", right_on="H3_Index", how="left"
)

occ_mask  = gdf_all["total_records"] > 0
unocc_mask = ~occ_mask

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — Priority Index with Component Scores Panel
# ══════════════════════════════════════════════════════════════════════════════
print("[1/5] Priority index …")

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(a)  Conservation Priority Index", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Occupied cells by priority index (Plasma colormap)
occ_gdf = gdf_all[occ_mask].copy()
norm_pri = mcolors.Normalize(vmin=0, vmax=1)
cmap_pri = plt.cm.plasma

occ_gdf["_c"] = occ_gdf["Priority_Index"].apply(
    lambda v: mcolors.to_hex(cmap_pri(norm_pri(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf.plot(ax=ax, color=occ_gdf["_c"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Label top-5 cells with rank number
top5_pri = occ_gdf[occ_gdf["Rank"] <= 5]
for _, row in top5_pri.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    rank = row.get("Rank", np.nan)
    if pd.notna(rank):
        ax.text(cx, cy, f"{int(rank)}", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=0.8, foreground="#3D0063")],
                zorder=6)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Colorbar
cax = fig.add_axes([0.81, 0.15, 0.015, 0.6])
cb = ColorbarBase(cax, cmap=cmap_pri, norm=norm_pri, orientation="vertical")
cb.set_label("Priority Index", fontsize=8.5)
cb.ax.tick_params(labelsize=7)

# Inset panel: component scores for top-5 cells
ax_ins = fig.add_axes([0.815, 0.65, 0.155, 0.25])
score_cols = ["score_richness", "score_diversity", "score_connectivity", "score_cooccurrence", "score_threatened"]
top5_idx = occ_gdf[occ_gdf["Rank"] <= 5].sort_values("Rank").index
if len(top5_idx) > 0:
    top_cell = occ_gdf.loc[top5_idx[0]]
    scores = [top_cell.get(col, 0) for col in score_cols]
    labels = ["S", "D", "C", "Co", "T"]
    colors_bar = ["#1565C0", "#42A5F5", "#90CAF9", "#BBDEFB", "#E3F2FD"]
    ax_ins.bar(labels, scores, color=colors_bar, alpha=0.8, edgecolor="#333", linewidth=0.5)
    ax_ins.set_ylim(0, 1)
    ax_ins.set_ylabel("Score", fontsize=6)
    ax_ins.set_title(f"Top cell scores", fontsize=7, pad=2)
    ax_ins.tick_params(labelsize=5.5)
    ax_ins.grid(True, axis="y", linewidth=0.25, alpha=0.5)

# Legend
handles = [
    s2_patch(),
    aoi_handle(),
    unocc_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Conservation Priority Index (0–1) for occupied cells · "
                 "Top-5 cells labelled with rank · Inset: component scores for rank #1 cell · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Prior_A_priority_index")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Priority Tier Categorical
# ══════════════════════════════════════════════════════════════════════════════
print("[2/5] Priority tier categorical …")

TIER_COLORS_PRI = {
    "CRITICAL": "#B71C1C",
    "HIGH": "#E65100",
    "MEDIUM": "#F57F17",
    "LOW": "#558B2F",
}

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(b)  Priority Tier", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Occupied cells by priority tier
occ_gdf = gdf_all[occ_mask].copy()
for tier, color in TIER_COLORS_PRI.items():
    subset = occ_gdf[occ_gdf["Priority_Tier"] == tier]
    if len(subset) == 0:
        continue
    subset.plot(ax=ax, color=color, edgecolor=BORDER_COLOR,
                linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Legend with counts
tier_counts = occ_gdf["Priority_Tier"].value_counts()
handles = [
    mpatches.Patch(facecolor="#B71C1C", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"CRITICAL (n = {tier_counts.get('CRITICAL', 0)})"),
    mpatches.Patch(facecolor="#E65100", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"HIGH (n = {tier_counts.get('HIGH', 0)})"),
    mpatches.Patch(facecolor="#F57F17", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"MEDIUM (n = {tier_counts.get('MEDIUM', 0)})"),
    mpatches.Patch(facecolor="#558B2F", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"LOW (n = {tier_counts.get('LOW', 0)})"),
    s2_patch(),
    aoi_handle(),
    unocc_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Priority tier for occupied cells · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Prior_B_priority_tier")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Survey Effort Tier
# ══════════════════════════════════════════════════════════════════════════════
print("[3/5] Survey effort tier …")

EFFORT_COLORS = {
    "High": "#1565C0",
    "Medium": "#42A5F5",
    "Low": "#EF9A9A",
}

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(c)  Survey Effort Tier", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Occupied cells by effort tier
occ_gdf = gdf_all[occ_mask].copy()
for tier, color in EFFORT_COLORS.items():
    subset = occ_gdf[occ_gdf["Effort_Tier"] == tier]
    if len(subset) == 0:
        continue
    subset.plot(ax=ax, color=color, edgecolor=BORDER_COLOR,
                linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Label Low tier cells with "!" warning
low_tier = occ_gdf[occ_gdf["Effort_Tier"] == "Low (<10 records)"]
for _, row in low_tier.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(cx, cy, "!", ha="center", va="center",
            fontsize=8, fontweight="bold", color="darkred",
            path_effects=[pe.withStroke(linewidth=0.7, foreground="white")],
            zorder=6)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Legend with counts
effort_counts = occ_gdf["Effort_Tier"].value_counts()
handles = [
    mpatches.Patch(facecolor="#1565C0", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"High (n = {effort_counts.get('High', 0)})"),
    mpatches.Patch(facecolor="#42A5F5", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"Medium (n = {effort_counts.get('Medium', 0)})"),
    mpatches.Patch(facecolor="#EF9A9A", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"Low (n = {effort_counts.get('Low', 0)})"),
    s2_patch(),
    aoi_handle(),
    unocc_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Survey effort tier for occupied cells · Low tier cells marked with ! · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Prior_C_survey_effort")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D — Flagship Species Presence
# ══════════════════════════════════════════════════════════════════════════════
print("[4/5] Flagship species presence …")

# Count flagship species per cell
flagship_cols = ["has_pant_pard", "has_hylo_molo", "has_nyct_java", "has_mani_java", "has_nisa_bart"]
gdf_all["flagship_count"] = (gdf_all[flagship_cols] == 1.0).sum(axis=1)

FLAGSHIP_COLORS = {
    0: "#E8E8E8",
    1: "#FFF9C4",
    2: "#FFCA28",
    3: "#FF8F00",
    4: "#E65100",
    5: "#B71C1C",
}

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(d)  Flagship Species Presence", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

# Plot occupied cells by flagship count
occ_gdf = gdf_all[occ_mask].copy()
for count in sorted(occ_gdf["flagship_count"].unique()):
    subset = occ_gdf[occ_gdf["flagship_count"] == count]
    color = FLAGSHIP_COLORS.get(int(count), "#FFFFFF")
    subset.plot(ax=ax, color=color, edgecolor=BORDER_COLOR,
                linewidth=BORDER_LW, alpha=OCC_ALPHA, zorder=3)

# Label cells with all 5 flagship species present
all5 = occ_gdf[occ_gdf["flagship_count"] == 5]
for _, row in all5.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(cx, cy, "✓", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#B71C1C",
            zorder=6)

draw_hex_base(ax, gdf_all[unocc_mask])
add_north_arrow(ax)
add_scalebar(ax)

# Legend
handles = [
    mpatches.Patch(facecolor="#B71C1C", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label="5 flagship species"),
    mpatches.Patch(facecolor="#E65100", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label="4 flagship species"),
    mpatches.Patch(facecolor="#FF8F00", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label="3 flagship species"),
    mpatches.Patch(facecolor="#FFCA28", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label="2 flagship species"),
    mpatches.Patch(facecolor="#FFF9C4", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
                   label="1 flagship species"),
    s2_patch(),
    aoi_handle(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, "Count of 5 flagship species per cell · "
                 "Javan Leopard, Javan Gibbon, Javan Slow Loris, Sunda Pangolin, Javan Hawk-Eagle · "
                 "Cells with all 5 marked with ✓ · Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_Prior_D_flagship_species")

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

# Panel A: Priority Index
ax_a = fig.get_axes()[0]
ax_a.set_title("(a)  Priority Index", fontweight="bold", fontsize=10, loc="left")
occ_gdf_a = gdf_all[occ_mask].copy()
norm_pri_a = mcolors.Normalize(vmin=0, vmax=1)
cmap_pri_a = plt.cm.plasma
occ_gdf_a["_c"] = occ_gdf_a["Priority_Index"].apply(
    lambda v: mcolors.to_hex(cmap_pri_a(norm_pri_a(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf_a.plot(ax=ax_a, color=occ_gdf_a["_c"].tolist(), edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)

# Panel B: Priority Tier
ax_b = fig.get_axes()[1]
ax_b.set_title("(b)  Priority Tier", fontweight="bold", fontsize=10, loc="left")
occ_gdf_b = gdf_all[occ_mask].copy()
for tier, color in TIER_COLORS_PRI.items():
    subset = occ_gdf_b[occ_gdf_b["Priority_Tier"] == tier]
    if len(subset) == 0:
        continue
    subset.plot(ax=ax_b, color=color, edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)

# Panel C: Survey Effort
ax_c = fig.get_axes()[2]
ax_c.set_title("(c)  Survey Effort Tier", fontweight="bold", fontsize=10, loc="left")
occ_gdf_c = gdf_all[occ_mask].copy()
for tier, color in EFFORT_COLORS.items():
    subset = occ_gdf_c[occ_gdf_c["Effort_Tier"] == tier]
    if len(subset) == 0:
        continue
    subset.plot(ax=ax_c, color=color, edgecolor="white", linewidth=1.0, alpha=0.85, zorder=3)

# Panel D: Flagship Species
ax_d = fig.get_axes()[3]
ax_d.set_title("(d)  Flagship Species Presence", fontweight="bold", fontsize=10, loc="left")
occ_gdf_d = gdf_all[occ_mask].copy()
for count in sorted(occ_gdf_d["flagship_count"].unique()):
    subset = occ_gdf_d[occ_gdf_d["flagship_count"] == count]
    color = FLAGSHIP_COLORS.get(int(count), "#FFFFFF")
    subset.plot(ax=ax_d, color=color, edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)

# ── Add north arrows and scale bars to all panels ──
for _ax in [ax_a, ax_b, ax_c, ax_d]:
    add_north_arrow(_ax)
    add_scalebar(_ax)

# ── Panel A legend ──
handles_a = [
    mpatches.Patch(facecolor="#7E0099", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="High priority (Plasma)"),
    mpatches.Patch(facecolor="#E8E8E8", edgecolor=BORDER_COLOR, alpha=0.6, label="Unoccupied cell"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax_a.legend(handles=handles_a, loc="upper left", fontsize=6.5, framealpha=0.90,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

# ── Panel B legend ──
handles_b = [
    mpatches.Patch(facecolor="#B71C1C", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="CRITICAL"),
    mpatches.Patch(facecolor="#E65100", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="HIGH"),
    mpatches.Patch(facecolor="#F57F17", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="MEDIUM"),
    mpatches.Patch(facecolor="#558B2F", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="LOW"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax_b.legend(handles=handles_b, loc="upper left", fontsize=6.5, framealpha=0.90,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

# ── Panel C legend ──
handles_c = [
    mpatches.Patch(facecolor="#1565C0", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="High effort"),
    mpatches.Patch(facecolor="#42A5F5", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Medium effort"),
    mpatches.Patch(facecolor="#EF9A9A", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="Low effort"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax_c.legend(handles=handles_c, loc="upper left", fontsize=6.5, framealpha=0.90,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

# ── Panel D legend ──
handles_d = [
    mpatches.Patch(facecolor="#B71C1C", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="5 flagship spp."),
    mpatches.Patch(facecolor="#E65100", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="4 flagship spp."),
    mpatches.Patch(facecolor="#FF8F00", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="3 flagship spp."),
    mpatches.Patch(facecolor="#FFCA28", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="2 flagship spp."),
    mpatches.Patch(facecolor="#FFF9C4", edgecolor=BORDER_COLOR, alpha=OCC_ALPHA, label="1 flagship spp."),
    Line2D([0], [0], color=AOI_COLOR, lw=1.2, linestyle="--", label="AOI boundary"),
]
ax_d.legend(handles=handles_d, loc="upper left", fontsize=6.5, framealpha=0.90,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.25)

save_fig(fig, OUT, "fig_Prior_E_composite")

print("\nAll priority figures complete!")
