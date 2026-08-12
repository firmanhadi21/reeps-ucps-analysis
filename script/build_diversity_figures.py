"""
build_diversity_figures.py
──────────────────────────
Generate four publication-ready static maps from the REEPS Diversity Map layers:
  Fig A — Species Richness (S)
  Fig B — Shannon Entropy (H′)
  Fig C — Simpson's Diversity Index (D)
  Fig D — Diversity Trend (Increasing / Stable / Decreasing)
  Fig E — 4-panel composite

Each figure uses the Sentinel-2 true-colour image (30 Sep 2024) as a natural
basemap reprojected to WGS84, with semi-transparent H3 hex overlays.

Each figure is saved as:
  • PNG at 300 DPI  (for manuscript embedding)
  • PDF (vector, for journal submission)

Output folder: <project root>/diversity_figures/
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
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import Polygon
from pathlib import Path
import h3
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import array_bounds

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = REEPS_BASE
BIO    = REEPS_BASE
OUT    = BIO / "diversity_figures"
OUT.mkdir(exist_ok=True)

S2_TIF = BIO / "Sentinel2_S2DR3_30Sep24_RGB.tif"

# ── Load data ──────────────────────────────────────────────────────────────────
df_div = pd.read_csv(BASE / "h3_diversity.csv")
df_h   = pd.read_csv(BASE / "h3_analysis.csv")
df_r   = pd.read_csv(BASE / "reeps_h3.csv")
aoi    = gpd.read_file(BIO / "aoi.gpkg").to_crs(4326)

# ── Build H3 polygon GeoDataFrame (all 149 cells) ────────────────────────────
def h3_to_polygon(cell):
    bnd = h3.cell_to_boundary(cell)   # [(lat, lon), ...]
    return Polygon([(lon, lat) for lat, lon in bnd])

all_cells = df_h["h3_index"].tolist()
geoms     = [h3_to_polygon(c) for c in all_cells]
gdf_all   = gpd.GeoDataFrame(df_h.copy(), geometry=geoms, crs="EPSG:4326")

# Merge diversity metrics
gdf_all = gdf_all.merge(
    df_div[["h3_index", "shannon_H", "simpson_D", "pielou_J",
            "berger_parker_BP", "dominant_species", "dominant_common",
            "diversity_trend", "diversity_trend_slope"]],
    on="h3_index", how="left"
)

# Occupied vs unoccupied masks
occ_mask  = gdf_all["total_records"] > 0
unocc_mask = ~occ_mask

# ── Load and warp Sentinel-2 basemap to WGS84 ─────────────────────────────────
# The source image is very large (21620×11120 px); we resample to ~3000×1500 px
# which is sufficient for 300 DPI figures while keeping memory usage reasonable.
print("Loading Sentinel-2 basemap (warping to WGS84, resampling) …")

TARGET_W = 3000   # target output width in pixels
DST_CRS  = "EPSG:4326"

with rasterio.open(S2_TIF) as src:
    # Compute the default WGS84 transform at native resolution
    transform_native, width_native, height_native = calculate_default_transform(
        src.crs, DST_CRS, src.width, src.height, *src.bounds
    )
    # Scale down to TARGET_W
    scale = TARGET_W / width_native
    dst_width  = TARGET_W
    dst_height = max(1, int(height_native * scale))

    # Build a scaled transform
    from rasterio.transform import from_bounds as transform_from_bounds
    lc_bounds_native = array_bounds(height_native, width_native, transform_native)
    dst_transform = transform_from_bounds(
        lc_bounds_native[0], lc_bounds_native[1],
        lc_bounds_native[2], lc_bounds_native[3],
        dst_width, dst_height
    )

    # Reproject bands 1-3 (R, G, B); band 4 is all-255 alpha, skip it
    rgb_wgs = np.zeros((3, dst_height, dst_width), dtype=np.uint8)
    for band_idx in range(1, 4):
        reproject(
            source=rasterio.band(src, band_idx),
            destination=rgb_wgs[band_idx - 1],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=DST_CRS,
            resampling=Resampling.bilinear,
        )

    lc_bounds = lc_bounds_native
    LC_EXTENT = (lc_bounds[0], lc_bounds[2], lc_bounds[1], lc_bounds[3])
    # (left, right, bottom, top) — matplotlib imshow convention

# Assemble float32 RGB image normalised to [0, 1]
lc_rgba = np.zeros((dst_height, dst_width, 4), dtype=np.float32)
lc_rgba[:, :, 0] = rgb_wgs[0] / 255.0
lc_rgba[:, :, 1] = rgb_wgs[1] / 255.0
lc_rgba[:, :, 2] = rgb_wgs[2] / 255.0
lc_rgba[:, :, 3] = 1.0   # fully opaque basemap

print(f"  Basemap size: {dst_width}×{dst_height} px, "
      f"WGS84 extent: [{LC_EXTENT[0]:.4f}, {LC_EXTENT[1]:.4f}] × "
      f"[{LC_EXTENT[2]:.4f}, {LC_EXTENT[3]:.4f}]")

# ── Publication style ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "DejaVu Sans"],
    "font.size":        9,
    "axes.titlesize":   11,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8.5,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.linewidth":   0.6,
})

# ── Map extent from AOI (with 10 % padding) ────────────────────────────────────
aoi_b  = aoi.total_bounds               # [minx, miny, maxx, maxy]
pad_x  = (aoi_b[2] - aoi_b[0]) * 0.10
pad_y  = (aoi_b[3] - aoi_b[1]) * 0.10
XMIN, YMIN = aoi_b[0] - pad_x, aoi_b[1] - pad_y
XMAX, YMAX = aoi_b[2] + pad_x, aoi_b[3] + pad_y

# ── Shared figure parameters ──────────────────────────────────────────────────
FIG_W, FIG_H  = 7.0, 5.4    # inches — A4 column-width compatible
UNOCC_COLOR   = "#E8E8E8"   # light grey border-only for unoccupied cells
UNOCC_ALPHA   = 0.25        # very transparent so basemap shows through
BORDER_COLOR  = "#555555"
BORDER_LW     = 0.35
AOI_COLOR     = "#E65100"
AOI_LW        = 1.8
OCC_ALPHA     = 0.65        # transparency for occupied cells


def setup_ax(fig, rect=(0.02, 0.02, 0.78, 0.96)):
    ax = fig.add_axes(rect)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.set_facecolor("none")   # transparent — basemap image handles background
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")
        spine.set_linewidth(0.5)
    # Faint grid graticule
    ax.grid(True, linestyle=":", linewidth=0.2, color="#BBBBBB", alpha=0.4, zorder=1)
    return ax


def add_basemap(ax):
    """Render the LC_2024.tif raster as an OSM-style background."""
    ax.imshow(
        lc_rgba,
        extent=LC_EXTENT,          # (left, right, bottom, top)
        origin="upper",
        aspect="equal",
        interpolation="bilinear",
        zorder=0,                  # behind everything
    )


def draw_base(ax):
    """Draw unoccupied hex borders and AOI boundary on top of basemap."""
    # Unoccupied cells: thin grey border, very transparent fill
    gdf_all[unocc_mask].plot(ax=ax, color=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                              linewidth=BORDER_LW, alpha=UNOCC_ALPHA, zorder=2)
    aoi.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=AOI_LW, zorder=5,
                      linestyle="--", label="AOI Boundary")


def add_north_arrow(ax):
    x, y = XMAX - (XMAX-XMIN)*0.065, YMIN + (YMAX-YMIN)*0.14
    ax.annotate("", xy=(x, y), xytext=(x, y - (YMAX-YMIN)*0.06),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
    ax.text(x, y + (YMAX-YMIN)*0.008, "N", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="black")


def add_scalebar(ax):
    lat_mid  = (YMIN + YMAX) / 2
    deg_per_km = 1 / (111.0 * np.cos(np.radians(abs(lat_mid))))
    scale_km   = 2
    bar_deg    = scale_km * deg_per_km
    bx = XMIN + (XMAX-XMIN)*0.05
    by = YMIN + (YMAX-YMIN)*0.05
    # White backing for readability over basemap
    ax.fill_between([bx - bar_deg*0.05, bx + bar_deg*1.05],
                    by - (YMAX-YMIN)*0.02, by + (YMAX-YMIN)*0.04,
                    color="white", alpha=0.7, zorder=8, linewidth=0)
    ax.plot([bx, bx + bar_deg], [by, by], color="black", lw=2,
            solid_capstyle="butt", zorder=9)
    ax.plot([bx, bx], [by - (YMAX-YMIN)*0.006, by + (YMAX-YMIN)*0.006],
            color="black", lw=1.5, zorder=9)
    ax.plot([bx+bar_deg, bx+bar_deg],
            [by - (YMAX-YMIN)*0.006, by + (YMAX-YMIN)*0.006],
            color="black", lw=1.5, zorder=9)
    ax.text(bx + bar_deg/2, by + (YMAX-YMIN)*0.014,
            f"{scale_km} km", ha="center", va="bottom", fontsize=7.5, zorder=9,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7,
                      edgecolor="none"))


def add_subtitle(ax, subtitle):
    ax.text(0.5, -0.01, subtitle, transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color="#444444",
            style="italic", wrap=True)


def s2_legend_patch():
    """Return a legend entry for the Sentinel-2 basemap."""
    return Line2D([0], [0], marker="s", color="none",
                  markerfacecolor="#5A7830", markeredgecolor="#444",
                  markeredgewidth=0.5, markersize=8,
                  label="Sentinel-2 basemap (30 Sep 2024)")


def save_fig(fig, stem):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"  ✓  {stem}.png / .pdf")
    plt.close(fig)


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
draw_base(ax)

occ_gdf = gdf_all[occ_mask].copy()
occ_gdf["color"] = occ_gdf["species_richness"].apply(
    lambda v: mcolors.to_hex(cmap_S(norm_S(max(1, v)))))
occ_gdf.plot(ax=ax, color=occ_gdf["color"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW + 0.1,
             alpha=OCC_ALPHA, zorder=3)

# Label cells with S ≥ 6
for _, row in occ_gdf[occ_gdf["species_richness"] >= 6].iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(cx, cy, str(int(row["species_richness"])),
            ha="center", va="center", fontsize=6.5, fontweight="bold", color="white",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
            zorder=6)

draw_base(ax)
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
    mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR, alpha=0.6,
                   label="Unoccupied cell"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.8, linestyle="--", label="AOI Boundary"),
    s2_legend_patch(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.4)

add_subtitle(ax, f"H3 Res-8 hexagonal grid · 38 occupied of 149 cells · "
                 f"UCPS Landscape, West Java · Survey years 2009–2026 · "
                 f"Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, "fig_A_species_richness")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Shannon Entropy H′
# ══════════════════════════════════════════════════════════════════════════════
print("[2/5] Shannon Entropy …")

H_vals = gdf_all.loc[occ_mask, "shannon_H"].dropna()
H_min, H_max = 0.0, H_vals.max()
cmap_H  = mpl.colormaps["Blues"]
norm_H  = mcolors.Normalize(vmin=0, vmax=H_max)

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(b)  Shannon Entropy (H′)", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_base(ax)

occ_gdf = gdf_all[occ_mask].copy()
occ_gdf["color"] = occ_gdf["shannon_H"].apply(
    lambda v: mcolors.to_hex(cmap_H(norm_H(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf.plot(ax=ax, color=occ_gdf["color"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW + 0.1,
             alpha=OCC_ALPHA, zorder=3)

# Label cells with H′ ≥ 1.5
for _, row in occ_gdf[occ_gdf["shannon_H"].fillna(0) >= 1.5].iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(cx, cy, f"{row['shannon_H']:.2f}",
            ha="center", va="center", fontsize=6, fontweight="bold", color="white",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="#333333")],
            zorder=6)

draw_base(ax)
add_north_arrow(ax)
add_scalebar(ax)

cax = fig.add_axes([0.82, 0.20, 0.025, 0.55])
cb  = ColorbarBase(cax, cmap=cmap_H, norm=norm_H, orientation="vertical")
cb.set_label("Shannon Entropy H′", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
H_ref = np.log(S_max)
if H_ref <= H_max:
    cb.ax.axhline(H_ref, color="#E65100", lw=1.2, linestyle="--")
    cb.ax.text(1.05, H_ref/H_max, f"ln({S_max})", transform=cb.ax.transAxes,
               fontsize=6.5, color="#E65100", va="center")

handles = [
    mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR, alpha=0.6,
                   label="Unoccupied cell"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.8, linestyle="--", label="AOI Boundary"),
    s2_legend_patch(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5)

add_subtitle(ax, f"H′ = 0 (single species) → {H_max:.3f} (max observed) · "
                 f"ln(S_max = {S_max}) = {np.log(S_max):.3f} · All survey years (2009–2026) · "
                 f"Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, "fig_B_shannon_entropy")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Simpson's Diversity Index D
# ══════════════════════════════════════════════════════════════════════════════
print("[3/5] Simpson's Diversity …")

D_vals = gdf_all.loc[occ_mask, "simpson_D"].dropna()
D_max  = D_vals.max()
cmap_D  = mpl.colormaps["YlGn"]
norm_D  = mcolors.Normalize(vmin=0, vmax=D_max)

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(c)  Simpson's Diversity Index (D)", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_base(ax)

occ_gdf = gdf_all[occ_mask].copy()
occ_gdf["color"] = occ_gdf["simpson_D"].apply(
    lambda v: mcolors.to_hex(cmap_D(norm_D(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf.plot(ax=ax, color=occ_gdf["color"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=BORDER_LW + 0.1,
             alpha=OCC_ALPHA, zorder=3)

for _, row in occ_gdf[occ_gdf["simpson_D"].fillna(0) >= 0.7].iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(cx, cy, f"{row['simpson_D']:.2f}",
            ha="center", va="center", fontsize=6, fontweight="bold", color="white",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="#333333")],
            zorder=6)

draw_base(ax)
add_north_arrow(ax)
add_scalebar(ax)

cax = fig.add_axes([0.82, 0.20, 0.025, 0.55])
cb  = ColorbarBase(cax, cmap=cmap_D, norm=norm_D, orientation="vertical")
cb.set_label("Simpson's D (1 − Σpᵢ²)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)

handles = [
    mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR, alpha=0.6,
                   label="Unoccupied cell"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.8, linestyle="--", label="AOI Boundary"),
    s2_legend_patch(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5)

# Inset: Pielou J vs Simpson D scatter
ax_ins = fig.add_axes([0.815, 0.78, 0.155, 0.155])
ax_ins.scatter(D_vals, gdf_all.loc[occ_mask, "pielou_J"].dropna(),
               s=12, alpha=0.7, color="#2E7D32", edgecolors="#1B5E20", linewidths=0.4)
ax_ins.set_xlabel("D", fontsize=6.5)
ax_ins.set_ylabel("J (Pielou)", fontsize=6.5)
ax_ins.tick_params(labelsize=6)
ax_ins.set_title("D vs J", fontsize=7, pad=2)
ax_ins.grid(True, linewidth=0.25, alpha=0.5)

add_subtitle(ax, f"D = 0 (single species) → {D_max:.3f} (max observed) · "
                 f"Inset: correlation with Pielou evenness (J) · "
                 f"Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, "fig_C_simpson_diversity")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D — Diversity Trend
# ══════════════════════════════════════════════════════════════════════════════
print("[4/5] Diversity Trend …")

TREND_COLORS = {
    "Increasing":  "#1B5E20",
    "Stable":      "#F57F17",
    "Decreasing":  "#B71C1C",
}

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = setup_ax(fig)
ax.set_title("(d)  Shannon Diversity Trend (2009–2026)", loc="left", pad=6,
             fontweight="bold", fontsize=11)

add_basemap(ax)
draw_base(ax)

occ_gdf = gdf_all[occ_mask].copy()
for trend, color in TREND_COLORS.items():
    subset = occ_gdf[occ_gdf["diversity_trend"] == trend]
    if len(subset) == 0:
        continue
    subset.plot(ax=ax, color=color, edgecolor=BORDER_COLOR,
                linewidth=BORDER_LW + 0.1, alpha=OCC_ALPHA, zorder=3)

# Label Decreasing cells with slope
decreasing = occ_gdf[occ_gdf["diversity_trend"] == "Decreasing"]
for _, row in decreasing.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    sl = row.get("diversity_trend_slope", np.nan)
    if pd.notna(sl):
        ax.text(cx, cy, f"{sl:.2f}", ha="center", va="center",
                fontsize=5.5, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=1.0, foreground="#7F0000")],
                zorder=6)

draw_base(ax)
add_north_arrow(ax)
add_scalebar(ax)

trend_counts = occ_gdf["diversity_trend"].value_counts()
handles = [
    mpatches.Patch(facecolor=TREND_COLORS["Increasing"], edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA,
                   label=f"Increasing  (n = {trend_counts.get('Increasing', 0)})"),
    mpatches.Patch(facecolor=TREND_COLORS["Stable"], edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA,
                   label=f"Stable         (n = {trend_counts.get('Stable', 0)})"),
    mpatches.Patch(facecolor=TREND_COLORS["Decreasing"], edgecolor=BORDER_COLOR,
                   alpha=OCC_ALPHA,
                   label=f"Decreasing  (n = {trend_counts.get('Decreasing', 0)})"),
    mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR, alpha=0.4,
                   label="Unoccupied cell"),
    Line2D([0], [0], color=AOI_COLOR, lw=1.8, linestyle="--", label="AOI Boundary"),
    s2_legend_patch(),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

# Inset: trend slope distribution (bar chart sorted)
slopes = occ_gdf["diversity_trend_slope"].dropna()
ax_ins = fig.add_axes([0.815, 0.69, 0.155, 0.20])
colors_hist = [TREND_COLORS["Decreasing"] if v < -0.01
               else (TREND_COLORS["Increasing"] if v > 0.01 else TREND_COLORS["Stable"])
               for v in slopes]
sorted_pairs  = sorted(zip(slopes, colors_hist))
sorted_slopes = [v for v, _ in sorted_pairs]
sorted_colors = [c for _, c in sorted_pairs]
ax_ins.bar(range(len(sorted_slopes)), sorted_slopes, color=sorted_colors,
           width=0.8, edgecolor="none", alpha=0.85)
ax_ins.axhline(0, color="black", lw=0.7, linestyle="--")
ax_ins.set_xlabel("Cells (sorted)", fontsize=6)
ax_ins.set_ylabel("Slope", fontsize=6)
ax_ins.tick_params(labelsize=5.5)
ax_ins.set_title("Trend slope dist.", fontsize=7, pad=2)
ax_ins.grid(True, axis="y", linewidth=0.25, alpha=0.5)

add_subtitle(ax, "Shannon H′ linear trend across survey periods · "
                 "Decreasing cells labelled with slope value · "
                 "Basemap: Sentinel-2 true colour (30 Sep 2024)")

save_fig(fig, "fig_D_diversity_trend")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE E — 4-panel composite (manuscript-ready)
# ══════════════════════════════════════════════════════════════════════════════
print("[5/5] 4-panel composite …")

fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.5), facecolor="white",
                          gridspec_kw={"hspace": 0.12, "wspace": 0.06})
fig.patch.set_facecolor("white")

PANEL_TITLES = [
    "(a)  Species Richness (S)",
    "(b)  Shannon Entropy (H′)",
    "(c)  Simpson's Diversity (D)",
    "(d)  Shannon Diversity Trend",
]

for ax in axes.flat:
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.set_facecolor("none")    # basemap handles background
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#AAAAAA")
        sp.set_linewidth(0.5)
    ax.grid(True, linestyle=":", linewidth=0.2, color="#BBBBBB", alpha=0.4, zorder=1)
    # Basemap
    ax.imshow(lc_rgba, extent=LC_EXTENT, origin="upper", aspect="equal",
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

# Panel B — Shannon
ax1 = axes[0, 1]
occ_gdf["_c"] = occ_gdf["shannon_H"].apply(
    lambda v: mcolors.to_hex(cmap_H(norm_H(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf.plot(ax=ax1, color=occ_gdf["_c"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)
cax1 = fig.add_axes([0.92, 0.565, 0.012, 0.36])
ColorbarBase(cax1, cmap=cmap_H, norm=norm_H, orientation="vertical").set_label("H′", fontsize=8)

# Panel C — Simpson
ax2 = axes[1, 0]
occ_gdf["_c"] = occ_gdf["simpson_D"].apply(
    lambda v: mcolors.to_hex(cmap_D(norm_D(v))) if pd.notna(v) else "#FFFFFF")
occ_gdf.plot(ax=ax2, color=occ_gdf["_c"].tolist(),
             edgecolor=BORDER_COLOR, linewidth=0.3, alpha=OCC_ALPHA, zorder=3)
cax2 = fig.add_axes([0.48, 0.08, 0.012, 0.36])
ColorbarBase(cax2, cmap=cmap_D, norm=norm_D, orientation="vertical").set_label("D", fontsize=8)

# Panel D — Trend
ax3 = axes[1, 1]
for trend, color in TREND_COLORS.items():
    sub = occ_gdf[occ_gdf["diversity_trend"] == trend]
    if len(sub):
        sub.plot(ax=ax3, color=color, edgecolor=BORDER_COLOR,
                 linewidth=0.3, alpha=OCC_ALPHA, zorder=3)
trend_handles = [
    mpatches.Patch(fc=c, ec=BORDER_COLOR, alpha=OCC_ALPHA,
                   label=f"{l} (n={trend_counts.get(l, 0)})")
    for l, c in TREND_COLORS.items()
]
trend_handles.append(mpatches.Patch(fc=UNOCC_COLOR, ec=BORDER_COLOR, alpha=0.4,
                                     label="Unoccupied"))
ax3.legend(handles=trend_handles, loc="upper left", fontsize=7, framealpha=0.85,
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
         "Alpha-diversity metrics for REEPS species across H3 Res-8 hexagonal cells, "
         "UCPS Landscape, West Java · Survey years: 2009–2026 · "
         "Basemap: Sentinel-2 true colour (30 Sep 2024)",
         ha="center", fontsize=8, color="#444444", style="italic")

save_fig(fig, "fig_E_diversity_composite")

print(f"\n✓ All figures saved to {OUT}")
print(f"  Files: {', '.join(p.name for p in sorted(OUT.iterdir()))}")
