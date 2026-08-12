"""
build_h3_static_figures.py
──────────────────────────
Generate four publication-ready static maps from the REEPS H3 Map layers:
  Fig A — Species Richness (S) - choropleth across all occupied cells
  Fig B — Total Records per cell - log-scale normalized distribution
  Fig C — Survey Year Coverage - count of years with detections
  Fig D — Recent Activity (2024 vs prior) - 2024-only, prior-only, both, undetected
  Fig E — 4-panel composite

Each figure uses the Sentinel-2 true-colour basemap (30 Sep 2024) as background
with semi-transparent H3 hex overlays (OCC_ALPHA=0.65).

Each figure is saved as:
  • PNG at 300 DPI  (for manuscript embedding)
  • PDF (vector, for journal submission)

Output folder: <project root>/static_figures/h3_map/
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

# ── Output directory ───────────────────────────────────────────────────────
OUT = STATIC_OUT / "h3_map"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading H3 data …")
df_h = pd.read_csv(BASE / "h3_analysis.csv")
df_r = pd.read_csv(BASE / "reeps_h3.csv")

# ── Build H3 polygon GeoDataFrame (all 149 cells) ────────────────────────
def h3_to_polygon_local(cell):
    bnd = h3.cell_to_boundary(cell)   # [(lat, lon), ...]
    return Polygon([(lon, lat) for lat, lon in bnd])

from shapely.geometry import Polygon

all_cells = df_h["h3_index"].tolist()
geoms     = [h3_to_polygon(c) for c in all_cells]
gdf_all   = gpd.GeoDataFrame(df_h.copy(), geometry=geoms, crs="EPSG:4326")

# Occupied vs unoccupied masks
occ_mask  = gdf_all["total_records"] > 0
unocc_mask = ~occ_mask

# ── Detect available year columns for coverage calculation ────────────────
year_cols = [col for col in df_h.columns if col.startswith('records_')]
years_list = sorted([int(col.split('_')[1]) for col in year_cols])
print(f"  Years available: {years_list}")

def count_years_with_detections(row):
    """Count how many years have total_records > 0 for this cell."""
    count = 0
    for year in years_list:
        col_name = f"records_{year}"
        if col_name in row and row[col_name] > 0:
            count += 1
    return count

gdf_all["year_coverage"] = gdf_all.apply(count_years_with_detections, axis=1)

print(f"  All {len(gdf_all)} cells loaded, {occ_mask.sum()} occupied")
print(f"  Year coverage range: {gdf_all['year_coverage'].min()}-{gdf_all['year_coverage'].max()}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — Species Richness
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Species Richness …")

S_max   = int(gdf_all["species_richness"].max())
cmap_S  = mpl.colormaps["YlOrRd"]
norm_S  = mcolors.Normalize(vmin=1, vmax=S_max)

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(a)  Species Richness (S)", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

occ_gdf = gdf_all[occ_mask].copy()
occ_gdf["color"] = occ_gdf["species_richness"].apply(
    lambda v: mcolors.to_hex(cmap_S(norm_S(max(1, v)))))
occ_gdf.plot(ax=ax, color=occ_gdf["color"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW + 0.1,
             alpha=OCC_ALPHA, zorder=3)

# Label cells with S >= 6
for _, row in occ_gdf[occ_gdf["species_richness"] >= 6].iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(cx, cy, str(int(row["species_richness"])),
            ha="center", va="center", fontsize=6.5, fontweight="bold", color="white",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
            zorder=6)

add_north_arrow(ax)
add_scalebar(ax)

# Colorbar
cax = fig.add_axes([0.82, 0.20, 0.025, 0.55])
cb  = ColorbarBase(cax, cmap=cmap_S, norm=norm_S,
                   orientation="vertical", ticks=range(1, S_max+1))
cb.set_label("Species Richness (S)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)

# Legend
handles = [
    unocc_handle(),
    aoi_handle(),
    s2_patch(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.4)

add_subtitle(ax, f"H3 Res-8 hexagonal grid · {occ_mask.sum()} occupied of {len(gdf_all)} cells · "
                 f"UCPS Landscape, West Java · Survey years 2009–2026 · "
                 f"Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_H3_A_richness")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Total Records per cell
# ══════════════════════════════════════════════════════════════════════════════
print("[2/5] Total Records …")

occ_records = gdf_all.loc[occ_mask, "total_records"]
rec_min, rec_max = occ_records.min(), occ_records.max()
rec_range = rec_max / rec_min if rec_min > 0 else 1.0

# Use log scale if range > 10x
if rec_range > 10:
    norm_R = mcolors.LogNorm(vmin=rec_min, vmax=rec_max)
else:
    norm_R = mcolors.Normalize(vmin=rec_min, vmax=rec_max)

cmap_R = mpl.colormaps["Blues"]

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(b)  Total Records per Cell", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

occ_gdf = gdf_all[occ_mask].copy()
occ_gdf["color"] = occ_gdf["total_records"].apply(
    lambda v: mcolors.to_hex(cmap_R(norm_R(max(1, v)))))
occ_gdf.plot(ax=ax, color=occ_gdf["color"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW + 0.1,
             alpha=OCC_ALPHA, zorder=3)

# Label top 5 cells with highest record count
top5 = occ_gdf.nlargest(5, "total_records")
for _, row in top5.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(cx, cy, str(int(row["total_records"])),
            ha="center", va="center", fontsize=6.5, fontweight="bold", color="white",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
            zorder=6)

add_north_arrow(ax)
add_scalebar(ax)

# Colorbar
cax = fig.add_axes([0.82, 0.20, 0.025, 0.55])
cb  = ColorbarBase(cax, cmap=cmap_R, norm=norm_R, orientation="vertical")
cb.set_label("Total Records", fontsize=8.5)
cb.ax.tick_params(labelsize=8)

# Legend
handles = [
    unocc_handle(),
    aoi_handle(),
    s2_patch(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.4)

scale_note = "log" if rec_range > 10 else "linear"
add_subtitle(ax, f"Survey effort across H3 cells · {scale_note.capitalize()} scale normalization · "
                 f"Range: {int(rec_min)}–{int(rec_max)} records · Top 5 cells labelled · "
                 f"Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_H3_B_records")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Survey Year Coverage
# ══════════════════════════════════════════════════════════════════════════════
print("[3/5] Year Coverage …")

max_years = gdf_all["year_coverage"].max()
cmap_C = mpl.colormaps["PuBu"]
norm_C = mcolors.Normalize(vmin=1, vmax=max_years)

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(c)  Survey Year Coverage", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_hex_base(ax, gdf_all[unocc_mask])

occ_gdf = gdf_all[occ_mask].copy()
occ_gdf["color"] = occ_gdf["year_coverage"].apply(
    lambda v: mcolors.to_hex(cmap_C(norm_C(v))))
occ_gdf.plot(ax=ax, color=occ_gdf["color"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW + 0.1,
             alpha=OCC_ALPHA, zorder=3)

# Label cells with >= 5 years coverage
for _, row in occ_gdf[occ_gdf["year_coverage"] >= 5].iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(cx, cy, str(int(row["year_coverage"])),
            ha="center", va="center", fontsize=6.5, fontweight="bold", color="white",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
            zorder=6)

add_north_arrow(ax)
add_scalebar(ax)

# Colorbar with integer ticks
cax = fig.add_axes([0.82, 0.20, 0.025, 0.55])
cb  = ColorbarBase(cax, cmap=cmap_C, norm=norm_C, orientation="vertical",
                   ticks=range(1, max_years+1))
cb.set_label("Survey Years", fontsize=8.5)
cb.ax.tick_params(labelsize=8)

# Legend
handles = [
    unocc_handle(),
    aoi_handle(),
    s2_patch(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.4)

add_subtitle(ax, f"Number of survey years with species detections per cell · "
                 f"Years checked: {years_list} · Cells with ≥5 years labelled · "
                 f"Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_H3_C_year_coverage")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D — Recent Activity (2024 vs Prior)
# ══════════════════════════════════════════════════════════════════════════════
print("[4/5] Recent Activity …")

# Determine cell status: categorize by 2024 presence vs prior years
def categorize_temporal_status(row):
    """
    Classify cells as:
    - "2024_only": species detected in 2024 but not in prior years
    - "prior_only": species detected in prior years but not in 2024
    - "both": species detected in both 2024 and prior years
    - "undetected": never detected
    """
    r_2024 = row.get("records_2024", 0)
    prior_years = [col for col in years_list if col != 2024]
    r_prior = sum([row.get(f"records_{y}", 0) for y in prior_years])

    if r_2024 == 0 and r_prior == 0:
        return "undetected"
    elif r_2024 > 0 and r_prior == 0:
        return "2024_only"
    elif r_2024 == 0 and r_prior > 0:
        return "prior_only"
    else:
        return "both"

gdf_all["temporal_status"] = gdf_all.apply(categorize_temporal_status, axis=1)

ACTIVITY_COLORS = {
    "2024_only":  "#2E7D32",   # green
    "prior_only": "#FF9800",   # amber
    "both":       "#1565C0",   # dark blue
    "undetected": "#E8E8E8",   # ghost/light grey
}

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(d)  Recent Activity (2024 vs Prior)", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)

# Draw all cells, starting with undetected
for status in ["undetected", "both", "prior_only", "2024_only"]:
    subset = gdf_all[gdf_all["temporal_status"] == status]
    if len(subset) == 0:
        continue
    if status == "undetected":
        subset.plot(ax=ax, color=ACTIVITY_COLORS[status], edgecolor=BORDER_COLOR,
                   linewidth=BORDER_LW, alpha=UNOCC_ALPHA, zorder=2)
    else:
        subset.plot(ax=ax, color=ACTIVITY_COLORS[status], edgecolor=BORDER_COLOR,
                   linewidth=BORDER_LW + 0.1, alpha=OCC_ALPHA, zorder=3)

aoi.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=AOI_LW, zorder=5, linestyle="--")

add_north_arrow(ax)
add_scalebar(ax)

# Legend
status_counts = gdf_all["temporal_status"].value_counts()
handles = [
    mpatches.Patch(facecolor=ACTIVITY_COLORS["2024_only"], edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA,
                   label=f"2024 only            (n = {status_counts.get('2024_only', 0)})"),
    mpatches.Patch(facecolor=ACTIVITY_COLORS["prior_only"], edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA,
                   label=f"Prior years only  (n = {status_counts.get('prior_only', 0)})"),
    mpatches.Patch(facecolor=ACTIVITY_COLORS["both"], edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA,
                   label=f"Both years         (n = {status_counts.get('both', 0)})"),
    mpatches.Patch(facecolor=ACTIVITY_COLORS["undetected"], edgecolor=BORDER_COLOR,
                   alpha=UNOCC_ALPHA,
                   label=f"Undetected       (n = {status_counts.get('undetected', 0)})"),
    aoi_handle(),
    s2_patch(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

add_subtitle(ax, f"Temporal distribution of species detections · "
                 f"Comparing 2024 survey year vs combined 2009–2022 period · "
                 f"Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, OUT, "fig_H3_D_recent_activity")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE E — 4-panel composite (manuscript-ready)
# ══════════════════════════════════════════════════════════════════════════════
print("[5/5] 4-panel composite …")

fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.5), facecolor="white",
                          gridspec_kw={"hspace": 0.12, "wspace": 0.06})
fig.patch.set_facecolor("white")

PANEL_TITLES = [
    "(a)  Species Richness (S)",
    "(b)  Total Records per Cell",
    "(c)  Survey Year Coverage",
    "(d)  Recent Activity (2024 vs Prior)",
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

# Panel A — Richness
ax0 = axes[0, 0]
occ_gdf = gdf_all[occ_mask].copy()
occ_gdf["_c"] = occ_gdf["species_richness"].apply(
    lambda v: mcolors.to_hex(cmap_S(norm_S(max(1, v)))))
occ_gdf.plot(ax=ax0, color=occ_gdf["_c"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)
cax0 = fig.add_axes([0.48, 0.565, 0.012, 0.36])
ColorbarBase(cax0, cmap=cmap_S, norm=norm_S, orientation="vertical",
             ticks=range(1, S_max+1)).set_label("S", fontsize=8)

# Panel B — Records
ax1 = axes[0, 1]
occ_gdf["_c"] = occ_gdf["total_records"].apply(
    lambda v: mcolors.to_hex(cmap_R(norm_R(max(1, v)))))
occ_gdf.plot(ax=ax1, color=occ_gdf["_c"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)
cax1 = fig.add_axes([0.92, 0.565, 0.012, 0.36])
ColorbarBase(cax1, cmap=cmap_R, norm=norm_R, orientation="vertical").set_label("Records", fontsize=8)

# Panel C — Year Coverage
ax2 = axes[1, 0]
occ_gdf["_c"] = occ_gdf["year_coverage"].apply(
    lambda v: mcolors.to_hex(cmap_C(norm_C(v))))
occ_gdf.plot(ax=ax2, color=occ_gdf["_c"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)
cax2 = fig.add_axes([0.48, 0.08, 0.012, 0.36])
ColorbarBase(cax2, cmap=cmap_C, norm=norm_C, orientation="vertical",
             ticks=range(1, max_years+1)).set_label("Years", fontsize=8)

# Panel D — Activity
ax3 = axes[1, 1]
for status in ["undetected", "both", "prior_only", "2024_only"]:
    sub = gdf_all[gdf_all["temporal_status"] == status]
    if len(sub):
        if status == "undetected":
            sub.plot(ax=ax3, color=ACTIVITY_COLORS[status], edgecolor=BORDER_COLOR,
                    linewidth=0.3, alpha=UNOCC_ALPHA, zorder=2)
        else:
            sub.plot(ax=ax3, color=ACTIVITY_COLORS[status], edgecolor=BORDER_COLOR,
                    linewidth=0.3, alpha=OCC_ALPHA, zorder=3)

activity_handles = [
    mpatches.Patch(fc=ACTIVITY_COLORS["2024_only"], ec=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"2024 only ({status_counts.get('2024_only', 0)})"),
    mpatches.Patch(fc=ACTIVITY_COLORS["prior_only"], ec=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"Prior ({status_counts.get('prior_only', 0)})"),
    mpatches.Patch(fc=ACTIVITY_COLORS["both"], ec=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"Both ({status_counts.get('both', 0)})"),
    mpatches.Patch(fc=ACTIVITY_COLORS["undetected"], ec=BORDER_COLOR, alpha=UNOCC_ALPHA,
                   label=f"Undetected ({status_counts.get('undetected', 0)})"),
]
ax3.legend(handles=activity_handles, loc="upper left", fontsize=7, framealpha=0.85,
           edgecolor="#CCC", handlelength=1.2)

# Titles, scalebars, north arrows on each panel
for i, (ax, title) in enumerate(zip(axes.flat, PANEL_TITLES)):
    ax.set_title(title, loc="left", pad=4, fontsize=10, fontweight="bold")
    if i >= 2:
        lat_mid = (YMIN+YMAX)/2
        deg2km  = 1/(111.0*np.cos(np.radians(abs(lat_mid))))
        blen    = 2*deg2km
        bx = XMIN + (XMAX-XMIN)*0.05
        by = YMIN + (YMAX-YMIN)*0.05
        ax.fill_between([bx - blen*0.05, bx + blen*1.05],
                        by - (YMAX-YMIN)*0.025, by + (YMAX-YMIN)*0.05,
                        color="white", alpha=0.7, zorder=8, linewidth=0)
        ax.plot([bx, bx+blen], [by, by], "k-", lw=2, solid_capstyle="butt", zorder=9)
        ax.text(bx+blen/2, by+(YMAX-YMIN)*0.015, "2 km",
                ha="center", fontsize=7, zorder=9)
    if i == 1:
        nx = XMAX-(XMAX-XMIN)*0.07
        ny = YMIN+(YMAX-YMIN)*0.14
        ax.annotate("", xy=(nx, ny), xytext=(nx, ny-(YMAX-YMIN)*0.07),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
        ax.text(nx, ny+(YMAX-YMIN)*0.01, "N",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

fig.text(0.5, 0.005,
         "H3 Res-8 hexagonal grid analysis across REEPS species distribution, "
         "UCPS Landscape, West Java · Survey years: 2009–2026 · "
         "Basemap: Sentinel-2 true colour (30 Sep 2024)",
         ha="center", fontsize=8, color="#444444", style="italic")

save_fig(fig, OUT, "fig_H3_E_composite")

print(f"\n✓ All H3 static figures saved to {OUT}")
print(f"  Files: {', '.join(sorted([p.name for p in OUT.glob('fig_H3_*.png')]))}")
