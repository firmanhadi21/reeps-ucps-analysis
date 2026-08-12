"""
build_revegetation_overlay.py
─────────────────────────────
Overlay revegetation plots (penanaman 2023–2024) with CPI Priority Tier
maps on the 1 km BMP grid.

Outputs:
  1. Revegetation overlay on CPI Priority Tier map
  2. Revegetation overlay on Conservation Model map
  3. Summary table of revegetation per grid cell and CPI tier
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
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import array_bounds, from_bounds as transform_from_bounds

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = REEPS_BASE
S2_TIF = BASE / "Sentinel2_S2DR3_30Sep24_RGB.tif"
GPKG = BASE / "REEPS_GridAnalyses_Square.gpkg"
AOI_PATH = BASE / "aoi.gpkg"
REVEG_SHP = BASE / "revegetation" / "Petak Revegetasi PLN - Perhutani" / "RTT_Tanaman_UCPS_2023-2024_Kirim_PLN.shp"
OUT_DIR = BASE / "static_figures" / "square_grid"

# ── Style ─────────────────────────────────────────────────────────────────
UNOCC_COLOR  = "#E8E8E8"
UNOCC_ALPHA  = 0.22
OCC_ALPHA    = 0.65
BORDER_COLOR = "#555555"
BORDER_LW    = 0.35
AOI_COLOR    = "#E65100"
AOI_LW       = 1.8
LABEL_STROKE = [pe.withStroke(linewidth=1.5, foreground="white")]

TIER_COLORS = {
    "CRITICAL": "#B71C1C",
    "HIGH":     "#E65100",
    "MEDIUM":   "#F57F17",
    "LOW":      "#558B2F",
}

MODEL_COLORS = {
    "Model 1": "#1B5E20",
    "Model 2": "#388E3C",
    "Model 3": "#66BB6A",
    "Model 4": "#A5D6A7",
    "Model 5": "#C8E6C9",
}

REVEG_COLORS = {
    2023: "#00BCD4",   # Cyan
    2024: "#FF4081",   # Pink
}
REVEG_EDGE = "#FFFFFF"
REVEG_ALPHA = 0.75
REVEG_LW = 1.0

mpl.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size":       9,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.linewidth":  0.6,
})

# ══════════════════════════════════════════════════════════════════════════
# Load basemap
# ══════════════════════════════════════════════════════════════════════════
print("Loading Sentinel-2 basemap ...")
_TARGET_W = 3000
with rasterio.open(S2_TIF) as _src:
    _t_native, _w_native, _h_native = calculate_default_transform(
        _src.crs, "EPSG:4326", _src.width, _src.height, *_src.bounds
    )
    _scale      = _TARGET_W / _w_native
    _dst_width  = _TARGET_W
    _dst_height = max(1, int(_h_native * _scale))
    _lc_bounds  = array_bounds(_h_native, _w_native, _t_native)
    _dst_transform = transform_from_bounds(
        _lc_bounds[0], _lc_bounds[1], _lc_bounds[2], _lc_bounds[3],
        _dst_width, _dst_height
    )
    _rgb = np.zeros((3, _dst_height, _dst_width), dtype=np.uint8)
    for _b in range(1, 4):
        reproject(
            source=rasterio.band(_src, _b),
            destination=_rgb[_b - 1],
            src_transform=_src.transform,
            src_crs=_src.crs,
            dst_transform=_dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
        )

S2_RGBA = np.zeros((_dst_height, _dst_width, 4), dtype=np.float32)
S2_RGBA[:, :, :3] = _rgb.transpose(1, 2, 0) / 255.0
S2_RGBA[:, :, 3] = 1.0
S2_EXTENT = (_lc_bounds[0], _lc_bounds[2], _lc_bounds[1], _lc_bounds[3])

# ══════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════
print("Loading data ...")
grid = gpd.read_file(GPKG, layer="grid_1km_priority")
aoi = gpd.read_file(AOI_PATH)
reveg = gpd.read_file(REVEG_SHP)

# Ensure consistent CRS
grid_4326 = grid.to_crs("EPSG:4326")
aoi_4326 = aoi.to_crs("EPSG:4326")
reveg_4326 = reveg.to_crs("EPSG:4326")
reveg_utm = reveg.to_crs("EPSG:32748")

print(f"  Revegetation plots: {len(reveg)} ({reveg_utm.geometry.area.sum()/10000:.1f} ha)")
print(f"  2023: {len(reveg[reveg['Tahun_UCPS']==2023])} plots ({reveg_utm[reveg_utm['Tahun_UCPS']==2023].geometry.area.sum()/10000:.1f} ha)")
print(f"  2024: {len(reveg[reveg['Tahun_UCPS']==2024])} plots ({reveg_utm[reveg_utm['Tahun_UCPS']==2024].geometry.area.sum()/10000:.1f} ha)")

# Map extent
_ab = aoi_4326.total_bounds
_px = (_ab[2] - _ab[0]) * 0.10
_py = (_ab[3] - _ab[1]) * 0.10
XMIN, YMIN = _ab[0] - _px, _ab[1] - _py
XMAX, YMAX = _ab[2] + _px, _ab[3] + _py

# ══════════════════════════════════════════════════════════════════════════
# Spatial analysis: which grid cells contain revegetation?
# ══════════════════════════════════════════════════════════════════════════
print("\nSpatial analysis ...")
# Use intersection (not just centroids) since plots may span cells
reveg_utm["reveg_area_ha"] = reveg_utm.geometry.area / 10000

# Intersect revegetation with grid
grid_utm = grid.copy()
grid_utm["grid_idx"] = range(len(grid_utm))
intersected = gpd.overlay(reveg_utm[["Tahun_UCPS", "ANAKPETAK", "KELASHUTAN", "Luas_Tan", "geometry"]],
                           grid_utm[["grid_idx", "Grid", "Priority_Tier", "Priority_Index",
                                     "species_richness", "flagship_count", "Model", "geometry"]],
                           how="intersection")
intersected["overlap_ha"] = intersected.geometry.area / 10000

print(f"  Intersection pieces: {len(intersected)}")

# Summary by grid cell
cell_summary = intersected.groupby(["Grid", "Priority_Tier", "Model"]).agg(
    n_plots=("ANAKPETAK", "nunique"),
    total_reveg_ha=("overlap_ha", "sum"),
    years=("Tahun_UCPS", lambda x: ", ".join(sorted(set(str(int(v)) for v in x)))),
).reset_index()

print("\n  Revegetation by grid cell and CPI tier:")
print(cell_summary.to_string(index=False))

# Summary by tier
tier_summary = intersected.groupby("Priority_Tier").agg(
    n_plots=("ANAKPETAK", "nunique"),
    total_ha=("overlap_ha", "sum"),
).reset_index()

print("\n  Revegetation area by CPI tier:")
for _, row in tier_summary.iterrows():
    tier = row["Priority_Tier"] if pd.notna(row["Priority_Tier"]) else "Unoccupied"
    print(f"    {tier}: {row['total_ha']:.2f} ha ({row['n_plots']} plots)")

# Unoccupied cells
unocc_reveg = intersected[intersected["Priority_Tier"].isna()]
if len(unocc_reveg):
    print(f"    Unoccupied cells: {unocc_reveg['overlap_ha'].sum():.2f} ha")


# ══════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════
def setup_ax(fig, rect=(0.02, 0.02, 0.78, 0.96)):
    ax = fig.add_axes(rect)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.set_facecolor("none")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#888888")
        sp.set_linewidth(0.5)
    ax.grid(True, linestyle=":", linewidth=0.2, color="#BBBBBB", alpha=0.4, zorder=1)
    return ax

def add_basemap(ax):
    ax.imshow(S2_RGBA, extent=S2_EXTENT, origin="upper",
              aspect="equal", interpolation="bilinear", zorder=0)

def draw_labels(ax, gdf, fontsize=5.0):
    for _, row in gdf.iterrows():
        label = row.get("Grid", "")
        if not label or label == "-" or pd.isna(label):
            continue
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color="#222222", alpha=0.85,
                path_effects=LABEL_STROKE, zorder=7)

def add_north_arrow(ax):
    x = XMAX - (XMAX - XMIN) * 0.065
    y = YMIN + (YMAX - YMIN) * 0.14
    ax.annotate("", xy=(x, y), xytext=(x, y - (YMAX - YMIN) * 0.06),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2), zorder=9)
    ax.text(x, y + (YMAX - YMIN) * 0.008, "N", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="black", zorder=9)

def add_scalebar(ax, km=2):
    lat_mid = (YMIN + YMAX) / 2
    deg_per_km = 1 / (111.0 * np.cos(np.radians(abs(lat_mid))))
    bar_deg = km * deg_per_km
    bx = XMIN + (XMAX - XMIN) * 0.05
    by = YMIN + (YMAX - YMIN) * 0.05
    ax.fill_between([bx - bar_deg * 0.05, bx + bar_deg * 1.05],
                    by - (YMAX - YMIN) * 0.02, by + (YMAX - YMIN) * 0.04,
                    color="white", alpha=0.7, zorder=8, linewidth=0)
    ax.plot([bx, bx + bar_deg], [by, by], color="black", lw=2,
            solid_capstyle="butt", zorder=9)
    for ex in [bx, bx + bar_deg]:
        ax.plot([ex, ex],
                [by - (YMAX - YMIN) * 0.006, by + (YMAX - YMIN) * 0.006],
                color="black", lw=1.5, zorder=9)
    ax.text(bx + bar_deg / 2, by + (YMAX - YMIN) * 0.014,
            f"{km} km", ha="center", va="bottom", fontsize=7.5, zorder=9,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                      alpha=0.7, edgecolor="none"))

def save_fig(fig, stem):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    print(f"  -> {stem}.png / .pdf")
    plt.close(fig)

def draw_reveg(ax, reveg_gdf):
    """Draw revegetation plots with year-based colouring."""
    for yr in [2023, 2024]:
        subset = reveg_gdf[reveg_gdf["Tahun_UCPS"] == yr]
        if len(subset):
            subset.plot(ax=ax, facecolor=REVEG_COLORS[yr],
                       edgecolor=REVEG_EDGE, linewidth=REVEG_LW,
                       alpha=REVEG_ALPHA, zorder=6)

def reveg_handles():
    handles = []
    for yr in [2023, 2024]:
        n = len(reveg[reveg["Tahun_UCPS"] == yr])
        area = reveg_utm[reveg_utm["Tahun_UCPS"] == yr].geometry.area.sum() / 10000
        handles.append(mpatches.Patch(
            facecolor=REVEG_COLORS[yr], edgecolor=REVEG_EDGE,
            alpha=REVEG_ALPHA, linewidth=0.8,
            label=f"Revegetation {yr} ({n} plots, {area:.0f} ha)"
        ))
    return handles


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Revegetation on CPI Priority Tier map
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Revegetation + CPI Priority Tier ...")
has_tier = grid_4326["Priority_Tier"].notna()

fig = plt.figure(figsize=(7.5, 6.0), facecolor="white")
ax = setup_ax(fig)
ax.set_title("Revegetation Plots (2023–2024) × CPI Priority Tier", loc="left",
             pad=6, fontweight="bold", fontsize=11)
add_basemap(ax)

# Unoccupied cells
grid_4326[~has_tier].plot(ax=ax, color=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                          linewidth=BORDER_LW, alpha=UNOCC_ALPHA, zorder=2)

# Priority tier cells
for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    subset = grid_4326[grid_4326["Priority_Tier"] == tier]
    if len(subset):
        subset.plot(ax=ax, color=TIER_COLORS[tier], edgecolor=BORDER_COLOR,
                    linewidth=BORDER_LW + 0.1, alpha=OCC_ALPHA, zorder=3)

aoi_4326.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=AOI_LW,
                        zorder=5, linestyle="--")

# Overlay revegetation plots
draw_reveg(ax, reveg_4326)

# Grid labels
draw_labels(ax, grid_4326, fontsize=5.0)

add_north_arrow(ax)
add_scalebar(ax)

# Legend
handles = []
for t in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
    n = len(grid_4326[grid_4326["Priority_Tier"] == t])
    handles.append(mpatches.Patch(
        facecolor=TIER_COLORS[t], edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
        label=f"{t} (n={n})"
    ))
handles.append(mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                               alpha=0.6, label="Unoccupied"))
handles += reveg_handles()
handles.append(Line2D([0], [0], color=AOI_COLOR, lw=1.8,
                       linestyle="--", label="AOI Boundary"))

ax.legend(handles=handles, loc="upper left", fontsize=6.0, framealpha=0.90,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

ax.text(0.5, -0.01,
        f"RTT revegetation plots (Perhutani–PLN, 2023–2024) overlaid on "
        f"CPI Priority Tier map · {len(reveg)} plots, {reveg_utm.geometry.area.sum()/10000:.0f} ha total · "
        f"UCPS Landscape, 1 km grid",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=7.5, color="#444444", style="italic")

save_fig(fig, "BMP_1km_I_reveg_x_tier")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Revegetation on Conservation Model map
# ══════════════════════════════════════════════════════════════════════════
print("Figure 2: Revegetation + Conservation Model ...")
has_model = grid_4326["Model"].notna() & (grid_4326["Model"] != "")

fig = plt.figure(figsize=(7.5, 6.0), facecolor="white")
ax = setup_ax(fig)
ax.set_title("Revegetation Plots (2023–2024) × Conservation Model", loc="left",
             pad=6, fontweight="bold", fontsize=11)
add_basemap(ax)

# No-model cells
grid_4326[~has_model].plot(ax=ax, color=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                           linewidth=BORDER_LW, alpha=UNOCC_ALPHA, zorder=2)

# Model cells
for model in ["Model 5", "Model 4", "Model 3", "Model 2", "Model 1"]:
    subset = grid_4326[grid_4326["Model"] == model]
    if len(subset):
        subset.plot(ax=ax, color=MODEL_COLORS[model], edgecolor=BORDER_COLOR,
                    linewidth=BORDER_LW + 0.1, alpha=OCC_ALPHA, zorder=3)

aoi_4326.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=AOI_LW,
                        zorder=5, linestyle="--")

# Overlay revegetation plots
draw_reveg(ax, reveg_4326)

# Grid labels
draw_labels(ax, grid_4326, fontsize=5.0)

add_north_arrow(ax)
add_scalebar(ax)

# Legend
handles = []
for m in ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"]:
    n = len(grid_4326[grid_4326["Model"] == m])
    handles.append(mpatches.Patch(
        facecolor=MODEL_COLORS[m], edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
        label=f"{m} (n={n})"
    ))
handles.append(mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                               alpha=0.6, label="No model assigned"))
handles += reveg_handles()
handles.append(Line2D([0], [0], color=AOI_COLOR, lw=1.8,
                       linestyle="--", label="AOI Boundary"))

ax.legend(handles=handles, loc="upper left", fontsize=6.0, framealpha=0.90,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

ax.text(0.5, -0.01,
        f"RTT revegetation plots (Perhutani–PLN) overlaid on BMP Conservation Model map · "
        f"UCPS Landscape, 1 km grid",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=7.5, color="#444444", style="italic")

save_fig(fig, "BMP_1km_J_reveg_x_model")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Side-by-side — Revegetation on Tier (left) vs Model (right)
# ══════════════════════════════════════════════════════════════════════════
print("Figure 3: Side-by-side composite ...")

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.5, 5.5), facecolor="white",
                                         gridspec_kw={"wspace": 0.04})

for ax, title in [(ax_left, "(a) Revegetation × CPI Priority Tier"),
                   (ax_right, "(b) Revegetation × Conservation Model")]:
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.set_facecolor("none")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#AAAAAA")
        sp.set_linewidth(0.5)
    ax.imshow(S2_RGBA, extent=S2_EXTENT, origin="upper",
              aspect="equal", interpolation="bilinear", zorder=0)
    ax.set_title(title, loc="left", pad=6, fontweight="bold", fontsize=10)

# Left: Priority Tier
grid_4326[~has_tier].plot(ax=ax_left, color=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                          linewidth=0.2, alpha=UNOCC_ALPHA, zorder=2)
for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    subset = grid_4326[grid_4326["Priority_Tier"] == tier]
    if len(subset):
        subset.plot(ax=ax_left, color=TIER_COLORS[tier], edgecolor=BORDER_COLOR,
                    linewidth=0.2, alpha=OCC_ALPHA, zorder=3)
aoi_4326.boundary.plot(ax=ax_left, color=AOI_COLOR, linewidth=1.2, zorder=5, linestyle="--")
draw_reveg(ax_left, reveg_4326)
draw_labels(ax_left, grid_4326, fontsize=3.5)
add_scalebar(ax_left)

tier_handles = [mpatches.Patch(fc=TIER_COLORS[t], ec=BORDER_COLOR, alpha=OCC_ALPHA,
                label=t) for t in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]]
tier_handles += reveg_handles()
ax_left.legend(handles=tier_handles, loc="upper left", fontsize=5.5, framealpha=0.85,
               edgecolor="#CCC", handlelength=1.0, labelspacing=0.25)

# Right: Conservation Model
grid_4326[~has_model].plot(ax=ax_right, color=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                           linewidth=0.2, alpha=UNOCC_ALPHA, zorder=2)
for model in ["Model 5", "Model 4", "Model 3", "Model 2", "Model 1"]:
    subset = grid_4326[grid_4326["Model"] == model]
    if len(subset):
        subset.plot(ax=ax_right, color=MODEL_COLORS[model], edgecolor=BORDER_COLOR,
                    linewidth=0.2, alpha=OCC_ALPHA, zorder=3)
aoi_4326.boundary.plot(ax=ax_right, color=AOI_COLOR, linewidth=1.2, zorder=5, linestyle="--")
draw_reveg(ax_right, reveg_4326)
draw_labels(ax_right, grid_4326, fontsize=3.5)
add_north_arrow(ax_right)

model_handles = [mpatches.Patch(fc=MODEL_COLORS[m], ec=BORDER_COLOR, alpha=OCC_ALPHA,
                 label=m) for m in ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"]]
model_handles += reveg_handles()
ax_right.legend(handles=model_handles, loc="upper left", fontsize=5.5, framealpha=0.85,
                edgecolor="#CCC", handlelength=1.0, labelspacing=0.25)

fig.text(0.5, -0.005,
         f"RTT Revegetation plots (Perhutani–PLN, 2023–2024): "
         f"{len(reveg)} plots, {reveg_utm.geometry.area.sum()/10000:.0f} ha · "
         f"Overlaid on CPI Priority Tier and BMP Conservation Model · "
         f"UCPS Landscape, 1 km grid",
         ha="center", fontsize=8, color="#444444", style="italic")

save_fig(fig, "BMP_1km_K_reveg_composite")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Summary bar chart — revegetation area by CPI tier and year
# ══════════════════════════════════════════════════════════════════════════
print("Figure 4: Summary bar chart ...")

# Compute areas by tier and year
tier_year = intersected.groupby(["Priority_Tier", "Tahun_UCPS"])["overlap_ha"].sum().reset_index()
# Add unoccupied
unocc_area = intersected[intersected["Priority_Tier"].isna()].groupby("Tahun_UCPS")["overlap_ha"].sum()
for yr, area in unocc_area.items():
    tier_year = pd.concat([tier_year, pd.DataFrame([{
        "Priority_Tier": "Unoccupied", "Tahun_UCPS": yr, "overlap_ha": area
    }])], ignore_index=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white",
                                gridspec_kw={"wspace": 0.35})

# Bar chart: area by tier
tier_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "Unoccupied"]
bar_data = {}
for tier in tier_order:
    subset = tier_year[tier_year["Priority_Tier"] == tier]
    bar_data[tier] = {
        2023: subset[subset["Tahun_UCPS"] == 2023]["overlap_ha"].sum(),
        2024: subset[subset["Tahun_UCPS"] == 2024]["overlap_ha"].sum(),
    }

x = np.arange(len(tier_order))
width = 0.35

bars_2023 = [bar_data[t].get(2023, 0) for t in tier_order]
bars_2024 = [bar_data[t].get(2024, 0) for t in tier_order]

bar_colors_2023 = REVEG_COLORS[2023]
bar_colors_2024 = REVEG_COLORS[2024]

b1 = ax1.bar(x - width/2, bars_2023, width, label="2023", color=bar_colors_2023,
             edgecolor="white", linewidth=0.8)
b2 = ax1.bar(x + width/2, bars_2024, width, label="2024", color=bar_colors_2024,
             edgecolor="white", linewidth=0.8)

ax1.set_xlabel("CPI Priority Tier", fontsize=10)
ax1.set_ylabel("Revegetation Area (ha)", fontsize=10)
ax1.set_title("(a) Revegetation Area by CPI Tier and Year", fontsize=11, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(tier_order, fontsize=9)
ax1.legend(fontsize=9)
ax1.grid(axis="y", linestyle=":", alpha=0.4)

# Add value labels on bars
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                     f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

# Pie chart: overall distribution
ax2.set_title("(b) Revegetation Distribution\nby CPI Priority Tier", fontsize=11, fontweight="bold")
pie_data = []
pie_labels = []
pie_colors = []
tier_color_map = {**TIER_COLORS, "Unoccupied": "#BDBDBD"}
for tier in tier_order:
    total = bar_data[tier].get(2023, 0) + bar_data[tier].get(2024, 0)
    if total > 0:
        pie_data.append(total)
        pie_labels.append(f"{tier}\n({total:.1f} ha)")
        pie_colors.append(tier_color_map.get(tier, "#999999"))

wedges, texts, autotexts = ax2.pie(
    pie_data, labels=pie_labels, colors=pie_colors,
    autopct="%1.1f%%", startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
for t in texts:
    t.set_fontsize(8)
for t in autotexts:
    t.set_fontsize(8)
    t.set_fontweight("bold")

fig.text(0.5, -0.02,
         f"Total revegetation: {reveg_utm.geometry.area.sum()/10000:.0f} ha across "
         f"{len(reveg)} plots (Perhutani–PLN RTT, 2023–2024) · "
         f"Intersected with CPI Priority Tier classification",
         ha="center", fontsize=8, color="#444444", style="italic")

plt.tight_layout()
save_fig(fig, "BMP_1km_L_reveg_summary")


# ══════════════════════════════════════════════════════════════════════════
# Save revegetation analysis to GPKG
# ══════════════════════════════════════════════════════════════════════════
print("\nSaving revegetation overlay to GPKG ...")
GPKG_OUT = BASE / "REEPS_GridAnalyses_Square.gpkg"

# Save revegetation polygons with grid cell info
reveg_with_grid = gpd.sjoin(reveg_utm, grid_utm[["grid_idx", "Grid", "Priority_Tier",
                            "Priority_Index", "Model", "geometry"]],
                            how="left", predicate="intersects")
_reveg_export = reveg_with_grid.drop(columns=["grid_idx", "index_right"], errors="ignore")
_reveg_export.to_file(GPKG_OUT, layer="revegetation_2023_2024", driver="GPKG")
print(f"  → layer 'revegetation_2023_2024': {len(_reveg_export)} records")

print("\nDone!")
