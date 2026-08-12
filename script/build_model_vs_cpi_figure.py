"""
build_model_vs_cpi_figure.py
────────────────────────────
Generate Conservation Model vs CPI Priority Tier comparison figures:
  1. Side-by-side map: Conservation Model (left) vs Priority Tier (right)
  2. Gap analysis map: highlight under-protected and over-allocated cells
  3. Cross-tabulation summary table figure
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
OUT_DIR = BASE / "static_figures" / "square_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
UNOCC_COLOR  = "#E8E8E8"
UNOCC_ALPHA  = 0.22
OCC_ALPHA    = 0.70
BORDER_COLOR = "#555555"
BORDER_LW    = 0.35
AOI_COLOR    = "#E65100"
AOI_LW       = 1.8
LABEL_STROKE = [pe.withStroke(linewidth=1.5, foreground="white")]

mpl.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size":       9,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.linewidth":  0.6,
})

# ── Colour palettes ──────────────────────────────────────────────────────
MODEL_COLORS = {
    "Model 1": "#1B5E20",   # Darkest green — highest protection
    "Model 2": "#388E3C",
    "Model 3": "#66BB6A",
    "Model 4": "#A5D6A7",
    "Model 5": "#C8E6C9",   # Lightest green — lowest protection
}

TIER_COLORS = {
    "CRITICAL": "#B71C1C",
    "HIGH":     "#E65100",
    "MEDIUM":   "#F57F17",
    "LOW":      "#558B2F",
}

GAP_COLORS = {
    "Under-protected": "#D50000",   # Red — high CPI but low model
    "Aligned":         "#2E7D32",   # Green — model matches CPI
    "Over-allocated":  "#1565C0",   # Blue — high model but low CPI
    "Survey gap":      "#9E9E9E",   # Grey — has model but no species data
}

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
print("Loading grid data ...")
grid = gpd.read_file(GPKG, layer="grid_1km_priority")
aoi = gpd.read_file(AOI_PATH)

grid_4326 = grid.to_crs("EPSG:4326")
aoi_4326 = aoi.to_crs("EPSG:4326")

# Map extent
_ab = aoi_4326.total_bounds
_px = (_ab[2] - _ab[0]) * 0.10
_py = (_ab[3] - _ab[1]) * 0.10
XMIN, YMIN = _ab[0] - _px, _ab[1] - _py
XMAX, YMAX = _ab[2] + _px, _ab[3] + _py

# Classify cells
has_model = grid_4326["Model"].notna() & (grid_4326["Model"] != "")
has_tier = grid_4326["Priority_Tier"].notna()
no_model = ~has_model


# ══════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════
def setup_panel(ax):
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.set_facecolor("none")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#AAAAAA")
        sp.set_linewidth(0.5)


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
    bx = XMIN + (XMAX - XMIN) * 0.85
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


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Side-by-side — Conservation Model vs Priority Tier
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Conservation Model vs Priority Tier (side-by-side) ...")

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.5, 5.5), facecolor="white",
                                         gridspec_kw={"wspace": 0.04})

# ── Left panel: Conservation Model ───────────────────────────────────────
setup_panel(ax_left)
add_basemap(ax_left)
ax_left.set_title("(a)  BMP Conservation Model", loc="left", pad=6,
                   fontweight="bold", fontsize=11)

# Unoccupied / no model cells
grid_4326[no_model].plot(ax=ax_left, color=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                         linewidth=BORDER_LW, alpha=UNOCC_ALPHA, zorder=2)

# Model-assigned cells
for model_name in ["Model 5", "Model 4", "Model 3", "Model 2", "Model 1"]:
    subset = grid_4326[grid_4326["Model"] == model_name]
    if len(subset):
        subset.plot(ax=ax_left, color=MODEL_COLORS[model_name],
                    edgecolor=BORDER_COLOR, linewidth=BORDER_LW + 0.1,
                    alpha=OCC_ALPHA, zorder=3)

aoi_4326.boundary.plot(ax=ax_left, color=AOI_COLOR, linewidth=AOI_LW,
                        zorder=5, linestyle="--")
draw_labels(ax_left, grid_4326, fontsize=4.5)
add_north_arrow(ax_left)
add_scalebar(ax_left)

# Legend
model_handles = []
for m in ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"]:
    n = len(grid_4326[grid_4326["Model"] == m])
    model_handles.append(mpatches.Patch(
        facecolor=MODEL_COLORS[m], edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
        label=f"{m} (n={n})"
    ))
model_handles.append(mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                                     alpha=0.6, label="No model assigned"))
model_handles.append(Line2D([0], [0], color=AOI_COLOR, lw=1.8,
                             linestyle="--", label="AOI Boundary"))
ax_left.legend(handles=model_handles, loc="upper left", fontsize=6.5,
               framealpha=0.88, edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

# ── Right panel: Priority Tier ───────────────────────────────────────────
setup_panel(ax_right)
add_basemap(ax_right)
ax_right.set_title("(b)  CPI Priority Tier (data-driven)", loc="left", pad=6,
                    fontweight="bold", fontsize=11)

# Unoccupied cells
grid_4326[~has_tier].plot(ax=ax_right, color=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                          linewidth=BORDER_LW, alpha=UNOCC_ALPHA, zorder=2)

# Tier cells
for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    subset = grid_4326[grid_4326["Priority_Tier"] == tier]
    if len(subset):
        subset.plot(ax=ax_right, color=TIER_COLORS[tier],
                    edgecolor=BORDER_COLOR, linewidth=BORDER_LW + 0.1,
                    alpha=OCC_ALPHA, zorder=3)

aoi_4326.boundary.plot(ax=ax_right, color=AOI_COLOR, linewidth=AOI_LW,
                        zorder=5, linestyle="--")
draw_labels(ax_right, grid_4326, fontsize=4.5)

# Legend
tier_handles = []
for t in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
    n = len(grid_4326[grid_4326["Priority_Tier"] == t])
    tier_handles.append(mpatches.Patch(
        facecolor=TIER_COLORS[t], edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
        label=f"{t} (n={n})"
    ))
tier_handles.append(mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                                    alpha=0.6, label="Unoccupied"))
tier_handles.append(Line2D([0], [0], color=AOI_COLOR, lw=1.8,
                            linestyle="--", label="AOI Boundary"))
ax_right.legend(handles=tier_handles, loc="upper left", fontsize=6.5,
                framealpha=0.88, edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)

fig.text(0.5, -0.005,
         "Comparison of pre-existing BMP Conservation Models (expert-assigned) "
         "with data-driven CPI Priority Tiers (species occurrence 2009–2024) · "
         "UCPS Landscape, West Java · 1 km grid",
         ha="center", fontsize=8, color="#444444", style="italic")

save_fig(fig, "BMP_1km_F_model_vs_tier")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Gap Analysis Map
# ══════════════════════════════════════════════════════════════════════════
print("Figure 2: Gap analysis map ...")

# Define model protection rank (1=highest, 5=lowest)
MODEL_RANK = {"Model 1": 1, "Model 2": 2, "Model 3": 3, "Model 4": 4, "Model 5": 5}
TIER_RANK = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}

def classify_gap(row):
    model = row["Model"]
    tier = row["Priority_Tier"]

    if pd.isna(model) or model == "":
        return "No model"
    if pd.isna(tier):
        return "Survey gap"

    m_rank = MODEL_RANK.get(model, 5)
    t_rank = TIER_RANK.get(tier, 4)

    # Under-protected: high CPI (CRITICAL/HIGH = rank 1-2) but low model (4-5)
    if t_rank <= 2 and m_rank >= 4:
        return "Under-protected"
    # Over-allocated: high model (1-2) but low CPI (MEDIUM/LOW = rank 3-4)
    if m_rank <= 2 and t_rank >= 3:
        return "Over-allocated"
    return "Aligned"

grid_4326["gap_class"] = grid_4326.apply(classify_gap, axis=1)

gap_counts = grid_4326["gap_class"].value_counts()
print(f"  Gap classification: {dict(gap_counts)}")

fig = plt.figure(figsize=(7.5, 6.0), facecolor="white")
ax = fig.add_axes([0.02, 0.02, 0.78, 0.96])
setup_panel(ax)
add_basemap(ax)
ax.set_title("BMP Model vs CPI — Gap Analysis", loc="left", pad=6,
             fontweight="bold", fontsize=11)

# No-model cells
grid_4326[grid_4326["gap_class"] == "No model"].plot(
    ax=ax, color=UNOCC_COLOR, edgecolor=BORDER_COLOR,
    linewidth=BORDER_LW, alpha=UNOCC_ALPHA, zorder=2)

# Draw each gap class
for gap_class in ["Survey gap", "Aligned", "Over-allocated", "Under-protected"]:
    subset = grid_4326[grid_4326["gap_class"] == gap_class]
    if len(subset):
        color = GAP_COLORS[gap_class]
        subset.plot(ax=ax, color=color, edgecolor=BORDER_COLOR,
                    linewidth=BORDER_LW + 0.1, alpha=OCC_ALPHA, zorder=3)

        # Add dual-label for mismatch cells: Model + Tier
        if gap_class in ("Under-protected", "Over-allocated"):
            for _, row in subset.iterrows():
                cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
                grid_lbl = row.get("Grid", "")
                model_short = row["Model"].replace("Model ", "M") if pd.notna(row["Model"]) else ""
                tier_short = row["Priority_Tier"][:4] if pd.notna(row["Priority_Tier"]) else ""
                ax.text(cx, cy + (YMAX - YMIN) * 0.008, grid_lbl,
                        ha="center", va="bottom", fontsize=4.5, fontweight="bold",
                        color="#222222", path_effects=LABEL_STROKE, zorder=7)
                ax.text(cx, cy - (YMAX - YMIN) * 0.008,
                        f"{model_short}/{tier_short}",
                        ha="center", va="top", fontsize=3.5, fontweight="bold",
                        color="white",
                        path_effects=[pe.withStroke(linewidth=1.2, foreground=color)],
                        zorder=7)

# Labels for aligned and survey gap cells (just grid label)
for gap_class in ["Aligned", "Survey gap"]:
    subset = grid_4326[grid_4326["gap_class"] == gap_class]
    for _, row in subset.iterrows():
        label = row.get("Grid", "")
        if not label or label == "-" or pd.isna(label):
            continue
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=4.5, fontweight="bold",
                color="#222222", alpha=0.85,
                path_effects=LABEL_STROKE, zorder=7)

aoi_4326.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=AOI_LW,
                        zorder=5, linestyle="--")
add_north_arrow(ax)
add_scalebar(ax)

# Legend
gap_handles = []
for cls, desc in [("Under-protected", "Under-protected (high CPI, low Model)"),
                  ("Over-allocated", "Over-allocated (high Model, low CPI)"),
                  ("Aligned", "Aligned (Model matches CPI)"),
                  ("Survey gap", "Survey gap (Model assigned, no species data)")]:
    n = gap_counts.get(cls, 0)
    gap_handles.append(mpatches.Patch(
        facecolor=GAP_COLORS[cls], edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
        label=f"{desc} (n={n})"
    ))
gap_handles.append(mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                                   alpha=0.6, label=f"No model assigned (n={gap_counts.get('No model', 0)})"))
gap_handles.append(Line2D([0], [0], color=AOI_COLOR, lw=1.8,
                           linestyle="--", label="AOI Boundary"))

ax.legend(handles=gap_handles, loc="upper left", fontsize=6.0,
          framealpha=0.90, edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.35)

ax.text(0.5, -0.01,
        "Under-protected: CRITICAL/HIGH CPI but Model 4–5 · "
        "Over-allocated: Model 1–2 but MEDIUM/LOW CPI · "
        "Mismatch cells show Model/Tier labels",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=7.5, color="#444444", style="italic")

save_fig(fig, "BMP_1km_G_gap_analysis")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Cross-tabulation heatmap
# ══════════════════════════════════════════════════════════════════════════
print("Figure 3: Cross-tabulation heatmap ...")

# Build cross-tab for cells with both Model and Tier
both = grid_4326[has_model & has_tier].copy()
ct = pd.crosstab(both["Model"], both["Priority_Tier"])
# Reorder
model_order = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"]
tier_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
ct = ct.reindex(index=[m for m in model_order if m in ct.index],
                columns=[t for t in tier_order if t in ct.columns], fill_value=0)

fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")

# Heatmap
cmap = plt.cm.YlOrRd
max_val = max(1, ct.values.max())
im = ax.imshow(ct.values, cmap=cmap, aspect="auto", vmin=0, vmax=max_val)

# Labels
ax.set_xticks(range(len(ct.columns)))
ax.set_xticklabels(ct.columns, fontsize=10, fontweight="bold")
ax.set_yticks(range(len(ct.index)))
ax.set_yticklabels(ct.index, fontsize=10, fontweight="bold")
ax.set_xlabel("CPI Priority Tier (data-driven)", fontsize=11, labelpad=10)
ax.set_ylabel("BMP Conservation Model (expert-assigned)", fontsize=11, labelpad=10)
ax.set_title("Cross-tabulation: Conservation Model vs CPI Priority Tier\n"
             "(n = cells with both Model and species data)",
             fontsize=12, fontweight="bold", pad=12)

# Cell values and highlight boxes
for i in range(len(ct.index)):
    for j in range(len(ct.columns)):
        val = ct.iloc[i, j]
        color = "white" if val > max_val * 0.6 else "black"
        ax.text(j, i, str(val), ha="center", va="center",
                fontsize=14, fontweight="bold", color=color)

        # Highlight mismatches
        m_rank = MODEL_RANK.get(ct.index[i], 5)
        t_rank = TIER_RANK.get(ct.columns[j], 4)
        if val > 0:
            if t_rank <= 2 and m_rank >= 4:  # under-protected
                rect = mpatches.FancyBboxPatch(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.05",
                    edgecolor="#D50000", linewidth=2.5, facecolor="none", zorder=5)
                ax.add_patch(rect)
            elif m_rank <= 2 and t_rank >= 3:  # over-allocated
                rect = mpatches.FancyBboxPatch(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.05",
                    edgecolor="#1565C0", linewidth=2.5, facecolor="none", zorder=5)
                ax.add_patch(rect)

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.04)
cbar.set_label("Number of cells", fontsize=9)

# Mismatch legend
legend_handles = [
    mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.05",
                             edgecolor="#D50000", linewidth=2.5, facecolor="none"),
    mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.05",
                             edgecolor="#1565C0", linewidth=2.5, facecolor="none"),
]
ax.legend(legend_handles,
          ["Under-protected (high CPI, low Model)",
           "Over-allocated (high Model, low CPI)"],
          loc="lower right", fontsize=7.5, framealpha=0.90,
          edgecolor="#CCC", handlelength=1.5)

# Add row/column totals
for i in range(len(ct.index)):
    row_total = ct.iloc[i].sum()
    ax.text(len(ct.columns) - 0.5 + 0.5, i, f"Σ={row_total}",
            ha="left", va="center", fontsize=8, color="#666666")

for j in range(len(ct.columns)):
    col_total = ct.iloc[:, j].sum()
    ax.text(j, len(ct.index) - 0.5 + 0.45, f"Σ={col_total}",
            ha="center", va="top", fontsize=8, color="#666666")

# Survey gap annotation
n_gap = gap_counts.get("Survey gap", 0)
ax.text(0.5, -0.18,
        f"Note: {n_gap} additional cells have Conservation Model assignments "
        f"but no species occurrence data (survey gaps). "
        f"Total cells with Models: {has_model.sum()}; with species data: {has_tier.sum()}.",
        transform=ax.transAxes, ha="center", fontsize=8,
        color="#444444", style="italic", wrap=True)

plt.tight_layout()
save_fig(fig, "BMP_1km_H_model_tier_crosstab")

print("\nDone!")
