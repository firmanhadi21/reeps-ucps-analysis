"""
build_square_grid_analyses.py
─────────────────────────────
Generate BMP priority maps and REEPS distribution maps using square grids
(1 km and 500 m) instead of H3 hexagonal cells.

Outputs:
  1. grid_cisokan_500m.gpkg        — 500 m square grid with labels
  2. BMP Priority Grid (1 km)      — 5 figures (A–E) with grid labels
  3. BMP Priority Grid (500 m)     — 5 figures (A–E) with grid labels
  4. REEPS Distribution Map (2025) — species distribution overview

All maps use Sentinel-2 true-colour basemap (30 Sep 2024).
Grid labels follow BMP convention: columns A–L, rows 0–8 (1 km);
500 m sub-cells use quadrant suffixes (a=SW, b=SE, c=NW, d=NE).
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
from shapely.geometry import box, Point
from pathlib import Path
from collections import Counter
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import array_bounds, from_bounds as transform_from_bounds

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = REEPS_BASE
S2_TIF = BASE / "Sentinel2_S2DR3_30Sep24_RGB.tif"
GRID_1KM = BASE / "grid_cisokan_1km.gpkg"
AOI_PATH = BASE / "aoi.gpkg"
MASTER_DB = BASE / "REEPS_Master_Database.gpkg"
OUT_DIR = BASE / "static_figures" / "square_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Map style constants ──────────────────────────────────────────────────
UNOCC_COLOR  = "#E8E8E8"
UNOCC_ALPHA  = 0.22
OCC_ALPHA    = 0.65
BORDER_COLOR = "#555555"
BORDER_LW    = 0.35
AOI_COLOR    = "#E65100"
AOI_LW       = 1.8
FIG_W, FIG_H = 7.5, 6.0

# Grid label style
LABEL_FONTSIZE     = 5.0      # single-panel maps
LABEL_FONTSIZE_SM  = 3.5      # composite panels
LABEL_COLOR        = "#222222"
LABEL_ALPHA        = 0.85
LABEL_STROKE_COLOR = "white"
LABEL_STROKE_WIDTH = 1.5

# Species palette (flagship species)
SP_COL = {
    "Panthera pardus melas":   "#C62828",
    "Hylobates moloch":        "#1565C0",
    "Nycticebus javanicus":    "#AD1457",
    "Manis javanica":          "#E65100",
    "Nisaetus bartelsi":       "#558B2F",
    "Trachypithecus auratus":  "#00838F",
    "Presbytis comata":        "#4527A0",
    "Prionailurus bengalensis":"#6D4C41",
    "Aonyx cinerea":           "#00695C",
    "Tragulus javanicus":      "#827717",
    "Hystrix javanica":        "#795548",
    "Macaca fascicularis":     "#607D8B",
}
ABBREV = {
    "Panthera pardus melas":   "Javan Leopard",
    "Hylobates moloch":        "Javan Gibbon",
    "Nycticebus javanicus":    "Javan Slow Loris",
    "Manis javanica":          "Sunda Pangolin",
    "Nisaetus bartelsi":       "Javan Hawk-Eagle",
    "Trachypithecus auratus":  "Javan Lutung",
    "Presbytis comata":        "Javan Surili",
    "Prionailurus bengalensis":"Leopard Cat",
    "Aonyx cinerea":           "Asian Small-clawed Otter",
    "Tragulus javanicus":      "Javan Mousedeer",
    "Hystrix javanica":        "Sunda Porcupine",
    "Macaca fascicularis":     "Long-tailed Macaque",
}

# Flagship (IUCN threatened) species for priority scoring
FLAGSHIP = [
    "Panthera pardus melas",    # CR
    "Hylobates moloch",         # EN
    "Nycticebus javanicus",     # EN → CR
    "Manis javanica",           # CR
    "Nisaetus bartelsi",        # EN
]

mpl.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size":       9,
    "axes.titlesize":  11,
    "axes.labelsize":  9,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.linewidth":  0.6,
})

# ══════════════════════════════════════════════════════════════════════════
# STEP 0: Load basemap (Sentinel-2 reprojected to EPSG:4326)
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
S2_RGBA[:, :, 0] = _rgb[0] / 255.0
S2_RGBA[:, :, 1] = _rgb[1] / 255.0
S2_RGBA[:, :, 2] = _rgb[2] / 255.0
S2_RGBA[:, :, 3] = 1.0
S2_EXTENT = (_lc_bounds[0], _lc_bounds[2], _lc_bounds[1], _lc_bounds[3])
print(f"  Basemap ready: {_dst_width}x{_dst_height} px")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Load data
# ══════════════════════════════════════════════════════════════════════════
print("\nLoading data ...")
grid_1km = gpd.read_file(GRID_1KM)  # EPSG:32748
aoi = gpd.read_file(AOI_PATH)       # EPSG:32748

# Occurrences — create proper geometry from Lat/Lon columns
occ_raw = gpd.read_file(MASTER_DB, layer="reeps_occurrences")
occ = gpd.GeoDataFrame(
    occ_raw.drop(columns=["geometry"]),
    geometry=[Point(lon, lat) for lon, lat in zip(occ_raw["Longitude"], occ_raw["Latitude"])],
    crs="EPSG:4326"
).to_crs("EPSG:32748")

# Filter out spatial outliers (points far from AOI)
aoi_bounds = aoi.total_bounds
buffer = 5000  # 5 km buffer
spatial_mask = (
    (occ.geometry.x >= aoi_bounds[0] - buffer) &
    (occ.geometry.x <= aoi_bounds[2] + buffer) &
    (occ.geometry.y >= aoi_bounds[1] - buffer) &
    (occ.geometry.y <= aoi_bounds[3] + buffer)
)
occ = occ[spatial_mask].copy()
print(f"  Occurrences: {len(occ)} records ({occ['Species'].nunique()} species)")
print(f"  1 km grid: {len(grid_1km)} cells")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Create 500 m grid with labels derived from parent 1 km cells
# ══════════════════════════════════════════════════════════════════════════
print("\nCreating 500 m grid ...")
bounds = grid_1km.total_bounds  # [xmin, ymin, xmax, ymax]
cell_size = 500  # meters

cells_500m = []
cell_id = 0
x = bounds[0]
while x < bounds[2]:
    y = bounds[1]
    while y < bounds[3]:
        cell = box(x, y, x + cell_size, y + cell_size)
        cells_500m.append({
            "cell_id": cell_id,
            "geometry": cell
        })
        cell_id += 1
        y += cell_size
    x += cell_size

grid_500m = gpd.GeoDataFrame(cells_500m, crs="EPSG:32748")

# Clip to AOI convex hull with buffer
aoi_hull = aoi.union_all().convex_hull.buffer(500)
grid_500m = grid_500m[grid_500m.geometry.intersects(aoi_hull)].copy()
grid_500m = grid_500m.reset_index(drop=True)

# Assign sub-labels from parent 1 km cell
# Each 1 km cell contains four 500 m sub-cells: a(SW), b(SE), c(NW), d(NE)
print("  Assigning 500 m sub-labels from parent 1 km grid ...")
grid_500m_centroids = grid_500m.copy()
grid_500m_centroids["centroid_geom"] = grid_500m_centroids.geometry.centroid
grid_500m_centroids_pts = grid_500m_centroids.set_geometry("centroid_geom")

joined_parent = gpd.sjoin(
    grid_500m_centroids_pts[["cell_id", "centroid_geom"]],
    grid_1km[["Grid", "geometry"]],
    how="left", predicate="within"
)

# Determine quadrant within parent cell
labels_500m = []
for _, row in grid_500m.iterrows():
    cid = row["cell_id"]
    match = joined_parent[joined_parent["cell_id"] == cid]
    if len(match) == 0 or match.iloc[0]["Grid"] == "-" or pd.isna(match.iloc[0]["Grid"]):
        labels_500m.append("-")
        continue

    parent_label = match.iloc[0]["Grid"]
    parent_idx = match.iloc[0]["index_right"]
    parent_geom = grid_1km.loc[parent_idx, "geometry"]
    parent_cx, parent_cy = parent_geom.centroid.x, parent_geom.centroid.y
    cell_cx, cell_cy = row.geometry.centroid.x, row.geometry.centroid.y

    if cell_cx < parent_cx:
        quad = "a" if cell_cy < parent_cy else "c"  # SW or NW
    else:
        quad = "b" if cell_cy < parent_cy else "d"  # SE or NE

    labels_500m.append(f"{parent_label}{quad}")

grid_500m["Grid"] = labels_500m
n_labeled = sum(1 for l in labels_500m if l != "-")
print(f"  500 m grid: {len(grid_500m)} cells, {n_labeled} labeled")

# Save 500m grid
grid_500m.to_file(BASE / "grid_cisokan_500m.gpkg", driver="GPKG")
print(f"  → grid_cisokan_500m.gpkg")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Spatial join & compute metrics for both grids
# ══════════════════════════════════════════════════════════════════════════
def compute_grid_metrics(grid, occ_pts, grid_name, id_col="cell_idx"):
    """Spatial join occurrences to grid and compute priority metrics."""
    print(f"\nComputing metrics for {grid_name} ...")

    g = grid.copy()
    g[id_col] = range(len(g))

    # Spatial join
    joined = gpd.sjoin(occ_pts, g[[id_col, "geometry"]], how="inner", predicate="within")

    print(f"  Records in grid: {len(joined)}")
    print(f"  Cells with records: {joined[id_col].nunique()}")

    # Per-cell metrics
    metrics = []
    for idx in range(len(g)):
        cell_data = joined[joined[id_col] == idx]
        n_records = len(cell_data)

        if n_records == 0:
            metrics.append({
                id_col: idx,
                "total_records": 0,
                "species_richness": 0,
                "shannon_h": 0.0,
                "simpson_d": 0.0,
                "n_threatened": 0,
                "n_years": 0,
                "species_list": "",
                "flagship_count": 0,
            })
            continue

        species = cell_data["Species"].value_counts()
        S = len(species)
        N = species.sum()

        # Shannon diversity
        props = species / N
        shannon = -np.sum(props * np.log(props)) if S > 1 else 0.0

        # Simpson diversity (1 - D)
        simpson = 1.0 - np.sum(props ** 2) if S > 1 else 0.0

        # Threatened/flagship species count
        n_threatened = sum(1 for sp in species.index if sp in FLAGSHIP)

        # Year coverage
        n_years = cell_data["Year"].dropna().nunique()

        metrics.append({
            id_col: idx,
            "total_records": int(N),
            "species_richness": int(S),
            "shannon_h": round(shannon, 4),
            "simpson_d": round(simpson, 4),
            "n_threatened": n_threatened,
            "n_years": n_years,
            "species_list": ", ".join(sorted(species.index)),
            "flagship_count": n_threatened,
        })

    df_metrics = pd.DataFrame(metrics)
    g = g.merge(df_metrics, on=id_col)

    # Compute normalized priority scores (only for occupied cells)
    occ_mask = g["total_records"] > 0
    if occ_mask.sum() > 0:
        occ_g = g.loc[occ_mask].copy()

        # Normalize each component to [0, 1]
        def norm_col(s):
            if s.max() == s.min():
                return pd.Series(0.5, index=s.index)
            return (s - s.min()) / (s.max() - s.min())

        occ_g["score_richness"] = norm_col(occ_g["species_richness"])
        occ_g["score_diversity"] = norm_col(occ_g["shannon_h"])
        occ_g["score_threatened"] = norm_col(occ_g["n_threatened"])

        # Connectivity: count occupied neighbors (cells sharing edge/corner)
        connectivity = []
        for _, row in g.iterrows():
            buffered = row.geometry.buffer(10)  # tiny buffer for adjacency
            n_occ_neighbors = sum(
                1 for _, other in occ_g.iterrows()
                if other[id_col] != row[id_col] and buffered.intersects(other.geometry)
            )
            connectivity.append(n_occ_neighbors)

        g["n_occ_neighbors"] = connectivity
        occ_g = occ_g.merge(g[[id_col, "n_occ_neighbors"]], on=id_col, how="left")
        occ_g["score_connectivity"] = norm_col(occ_g["n_occ_neighbors"])

        # Co-occurrence: proportion of cells where >1 species co-occurs
        occ_g["score_cooccurrence"] = (occ_g["species_richness"] > 1).astype(float)

        # Priority Index (weighted mean)
        weights = {
            "score_richness": 0.25,
            "score_diversity": 0.20,
            "score_connectivity": 0.20,
            "score_cooccurrence": 0.15,
            "score_threatened": 0.20,
        }
        occ_g["Priority_Index"] = sum(
            occ_g[col] * w for col, w in weights.items()
        )

        # Rank and Tier
        occ_g["Rank"] = occ_g["Priority_Index"].rank(ascending=False, method="min").astype(int)

        def assign_tier(pi):
            if pi >= 0.70:
                return "CRITICAL"
            elif pi >= 0.50:
                return "HIGH"
            elif pi >= 0.30:
                return "MEDIUM"
            else:
                return "LOW"

        occ_g["Priority_Tier"] = occ_g["Priority_Index"].apply(assign_tier)

        # Merge back
        score_cols = ["score_richness", "score_diversity", "score_connectivity",
                      "score_cooccurrence", "score_threatened", "Priority_Index",
                      "Rank", "Priority_Tier"]
        for col in score_cols:
            g[col] = np.nan
        g.loc[occ_mask, score_cols] = occ_g[score_cols].values

    tier_counts = g.loc[occ_mask, "Priority_Tier"].value_counts()
    print(f"  Priority tiers: {dict(tier_counts)}")

    return g


# Compute metrics for both grids
grid_1km_metrics = compute_grid_metrics(grid_1km, occ, "1 km grid")
grid_500m_metrics = compute_grid_metrics(grid_500m, occ, "500 m grid")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3b: Save analysis results as GPKG
# ══════════════════════════════════════════════════════════════════════════
print("\nSaving analysis results to GPKG ...")

GPKG_OUT = BASE / "REEPS_GridAnalyses_Square.gpkg"

# Select columns to export (drop internal index columns)
_export_cols_1km = [c for c in grid_1km_metrics.columns if c != "cell_idx"]
_export_cols_500m = [c for c in grid_500m_metrics.columns if c != "cell_idx"]

grid_1km_metrics[_export_cols_1km].to_file(GPKG_OUT, layer="grid_1km_priority", driver="GPKG")
grid_500m_metrics[_export_cols_500m].to_file(GPKG_OUT, layer="grid_500m_priority", driver="GPKG")

# Also save occurrences joined to 1km grid for reference
occ_with_grid = gpd.sjoin(occ, grid_1km_metrics[["cell_idx", "Grid", "geometry"]],
                          how="left", predicate="within")
_occ_export = occ_with_grid.drop(columns=["cell_idx", "index_right"], errors="ignore")
_occ_export.to_file(GPKG_OUT, layer="occurrences_gridded", driver="GPKG")

print(f"  → {GPKG_OUT.name}")
print(f"    layer 'grid_1km_priority':    {len(grid_1km_metrics)} cells")
print(f"    layer 'grid_500m_priority':   {len(grid_500m_metrics)} cells")
print(f"    layer 'occurrences_gridded':  {len(_occ_export)} records")


# ══════════════════════════════════════════════════════════════════════════
# HELPER: Map drawing functions
# ══════════════════════════════════════════════════════════════════════════

# Convert grids to EPSG:4326 for plotting with basemap
grid_1km_4326 = grid_1km_metrics.to_crs("EPSG:4326")
grid_500m_4326 = grid_500m_metrics.to_crs("EPSG:4326")
aoi_4326 = aoi.to_crs("EPSG:4326")

# Map extent from AOI
_ab = aoi_4326.total_bounds
_px = (_ab[2] - _ab[0]) * 0.10
_py = (_ab[3] - _ab[1]) * 0.10
XMIN, YMIN = _ab[0] - _px, _ab[1] - _py
XMAX, YMAX = _ab[2] + _px, _ab[3] + _py


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


def draw_grid_base(ax, gdf_unocc):
    gdf_unocc.plot(ax=ax, color=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                   linewidth=BORDER_LW, alpha=UNOCC_ALPHA, zorder=2)
    aoi_4326.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=AOI_LW,
                           zorder=5, linestyle="--")


def draw_grid_labels(ax, gdf, fontsize=None, skip_unocc=False):
    """Draw grid cell labels (A1, B2, etc.) at cell centroids."""
    if fontsize is None:
        fontsize = LABEL_FONTSIZE
    for _, row in gdf.iterrows():
        label = row.get("Grid", "")
        if not label or label == "-" or pd.isna(label):
            continue
        if skip_unocc and row.get("total_records", 0) == 0:
            continue
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color=LABEL_COLOR, alpha=LABEL_ALPHA,
                path_effects=[pe.withStroke(linewidth=LABEL_STROKE_WIDTH,
                                           foreground=LABEL_STROKE_COLOR)],
                zorder=7)


def add_north_arrow(ax):
    x = XMAX - (XMAX - XMIN) * 0.065
    y = YMIN + (YMAX - YMIN) * 0.14
    ax.annotate("", xy=(x, y), xytext=(x, y - (YMAX - YMIN) * 0.06),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2), zorder=9)
    ax.text(x, y + (YMAX - YMIN) * 0.008, "N", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="black", zorder=9)


def add_scalebar(ax, km=2):
    lat_mid    = (YMIN + YMAX) / 2
    deg_per_km = 1 / (111.0 * np.cos(np.radians(abs(lat_mid))))
    bar_deg    = km * deg_per_km
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


def add_subtitle(ax, text):
    ax.text(0.5, -0.01, text, transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color="#444444",
            style="italic", wrap=True)


def save_fig(fig, stem):
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"  -> {stem}.png / .pdf")
    plt.close(fig)


def s2_patch():
    return Line2D([0], [0], marker="s", color="none",
                  markerfacecolor="#5A7830", markeredgecolor="#444",
                  markeredgewidth=0.5, markersize=8,
                  label="Sentinel-2 basemap (30 Sep 2024)")

def aoi_handle():
    return Line2D([0], [0], color=AOI_COLOR, lw=1.8,
                  linestyle="--", label="AOI Boundary")

def unocc_handle():
    return mpatches.Patch(facecolor=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                          alpha=0.6, label="Unoccupied cell")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: BMP PRIORITY MAPS (for each grid resolution)
# ══════════════════════════════════════════════════════════════════════════

TIER_COLORS = {
    "CRITICAL": "#B71C1C",
    "HIGH":     "#E65100",
    "MEDIUM":   "#F57F17",
    "LOW":      "#558B2F",
}

def make_priority_maps(gdf, res_label, prefix):
    """Generate 5 priority map figures for a given grid resolution."""
    occ_mask = gdf["total_records"] > 0
    unocc_mask = ~occ_mask
    n_occ = occ_mask.sum()
    n_total = len(gdf)

    # Choose label font size based on grid resolution
    is_500m = "500" in res_label
    lbl_fs = 3.5 if is_500m else LABEL_FONTSIZE

    # ── Fig A: Species Richness ──────────────────────────────────────────
    print(f"\n[{prefix}] Fig A: Species Richness ({res_label}) ...")
    S_max = max(1, int(gdf["species_richness"].max()))
    cmap_S = mpl.colormaps["YlOrRd"]
    norm_S = mcolors.Normalize(vmin=1, vmax=S_max)

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax = setup_ax(fig)
    ax.set_title(f"(a)  Species Richness — {res_label} grid", loc="left", pad=6,
                 fontweight="bold", fontsize=11)
    add_basemap(ax)
    draw_grid_base(ax, gdf[unocc_mask])

    occ_gdf = gdf[occ_mask].copy()
    occ_gdf["color"] = occ_gdf["species_richness"].apply(
        lambda v: mcolors.to_hex(cmap_S(norm_S(max(1, v)))))
    occ_gdf.plot(ax=ax, color=occ_gdf["color"].tolist(),
                 edgecolor=BORDER_COLOR, linewidth=BORDER_LW + 0.1,
                 alpha=OCC_ALPHA, zorder=3)

    # Draw grid labels on ALL cells
    draw_grid_labels(ax, gdf, fontsize=lbl_fs)

    add_north_arrow(ax)
    add_scalebar(ax)

    cax = fig.add_axes([0.82, 0.20, 0.025, 0.55])
    cb = ColorbarBase(cax, cmap=cmap_S, norm=norm_S, orientation="vertical",
                      ticks=range(1, S_max + 1))
    cb.set_label("Species Richness (S)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    handles = [unocc_handle(), aoi_handle(), s2_patch()]
    ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
              edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.4)
    add_subtitle(ax, f"{res_label} square grid · {n_occ} occupied of {n_total} cells · "
                     f"UCPS Landscape, West Java · Survey years 2009–2024 · "
                     f"Basemap: Sentinel-2 (30 Sep 2024)")
    save_fig(fig, f"{prefix}_A_richness")

    # ── Fig B: Conservation Priority Index ───────────────────────────────
    print(f"[{prefix}] Fig B: Priority Index ({res_label}) ...")
    cmap_pri = plt.cm.plasma
    norm_pri = mcolors.Normalize(vmin=0, vmax=1)

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax = setup_ax(fig)
    ax.set_title(f"(b)  Conservation Priority Index — {res_label} grid", loc="left", pad=6,
                 fontweight="bold", fontsize=11)
    add_basemap(ax)
    draw_grid_base(ax, gdf[unocc_mask])

    occ_gdf = gdf[occ_mask].copy()
    occ_gdf["color"] = occ_gdf["Priority_Index"].apply(
        lambda v: mcolors.to_hex(cmap_pri(norm_pri(v))) if pd.notna(v) else "#FFFFFF")
    occ_gdf.plot(ax=ax, color=occ_gdf["color"].tolist(),
                 edgecolor=BORDER_COLOR, linewidth=BORDER_LW + 0.1,
                 alpha=OCC_ALPHA, zorder=3)

    # Draw grid labels on ALL cells
    draw_grid_labels(ax, gdf, fontsize=lbl_fs)

    add_north_arrow(ax)
    add_scalebar(ax)

    cax = fig.add_axes([0.82, 0.20, 0.025, 0.55])
    cb = ColorbarBase(cax, cmap=cmap_pri, norm=norm_pri, orientation="vertical")
    cb.set_label("Priority Index", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    handles = [unocc_handle(), aoi_handle(), s2_patch()]
    ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
              edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.4)
    add_subtitle(ax, f"Weighted composite of richness, diversity, connectivity, "
                     f"co-occurrence & threatened spp. · Top 5 cells labelled")
    save_fig(fig, f"{prefix}_B_priority_index")

    # ── Fig C: Priority Tier ─────────────────────────────────────────────
    print(f"[{prefix}] Fig C: Priority Tier ({res_label}) ...")
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax = setup_ax(fig)
    ax.set_title(f"(c)  Priority Tier — {res_label} grid", loc="left", pad=6,
                 fontweight="bold", fontsize=11)
    add_basemap(ax)
    draw_grid_base(ax, gdf[unocc_mask])

    occ_gdf = gdf[occ_mask].copy()
    for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        subset = occ_gdf[occ_gdf["Priority_Tier"] == tier]
        if len(subset):
            subset.plot(ax=ax, color=TIER_COLORS[tier], edgecolor=BORDER_COLOR,
                        linewidth=BORDER_LW + 0.1, alpha=OCC_ALPHA, zorder=3)

    # Draw grid labels on ALL cells
    draw_grid_labels(ax, gdf, fontsize=lbl_fs)

    add_north_arrow(ax)
    add_scalebar(ax)

    tier_counts = occ_gdf["Priority_Tier"].value_counts()
    handles = []
    for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        n = tier_counts.get(tier, 0)
        handles.append(mpatches.Patch(
            facecolor=TIER_COLORS[tier], edgecolor=BORDER_COLOR, alpha=OCC_ALPHA,
            label=f"{tier} (n={n})"
        ))
    handles += [unocc_handle(), aoi_handle(), s2_patch()]
    ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
              edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)
    add_subtitle(ax, f"Priority tiers: CRITICAL (≥0.70), HIGH (0.50–0.69), "
                     f"MEDIUM (0.30–0.49), LOW (<0.30)")
    save_fig(fig, f"{prefix}_C_priority_tier")

    # ── Fig D: Flagship Species Presence ─────────────────────────────────
    print(f"[{prefix}] Fig D: Flagship Species ({res_label}) ...")
    FLAGSHIP_COLORS = {
        0: "#E8E8E8", 1: "#FFF9C4", 2: "#FFCA28",
        3: "#FF8F00",  4: "#E65100", 5: "#B71C1C",
    }

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax = setup_ax(fig)
    ax.set_title(f"(d)  Flagship Species Presence — {res_label} grid", loc="left", pad=6,
                 fontweight="bold", fontsize=11)
    add_basemap(ax)
    draw_grid_base(ax, gdf[unocc_mask])

    occ_gdf = gdf[occ_mask].copy()
    for count in sorted(occ_gdf["flagship_count"].unique()):
        subset = occ_gdf[occ_gdf["flagship_count"] == count]
        color = FLAGSHIP_COLORS.get(int(count), "#FFFFFF")
        subset.plot(ax=ax, color=color, edgecolor=BORDER_COLOR,
                    linewidth=BORDER_LW + 0.1, alpha=OCC_ALPHA, zorder=3)

    # Draw grid labels on ALL cells
    draw_grid_labels(ax, gdf, fontsize=lbl_fs)

    add_north_arrow(ax)
    add_scalebar(ax)

    max_fs = int(occ_gdf["flagship_count"].max())
    handles = []
    for i in range(max_fs, 0, -1):
        n = len(occ_gdf[occ_gdf["flagship_count"] == i])
        handles.append(mpatches.Patch(
            facecolor=FLAGSHIP_COLORS.get(i, "#FFF"), edgecolor=BORDER_COLOR,
            alpha=OCC_ALPHA, label=f"{i} flagship spp. (n={n})"
        ))
    handles += [unocc_handle(), aoi_handle(), s2_patch()]
    ax.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.88,
              edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.3)
    add_subtitle(ax, "Flagship: Javan Leopard, Javan Gibbon, Javan Slow Loris, "
                     "Sunda Pangolin, Javan Hawk-Eagle")
    save_fig(fig, f"{prefix}_D_flagship")

    # ── Fig E: 4-panel composite ─────────────────────────────────────────
    print(f"[{prefix}] Fig E: 4-panel composite ({res_label}) ...")
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 11.0), facecolor="white",
                              gridspec_kw={"hspace": 0.12, "wspace": 0.06})

    PANEL_TITLES = [
        f"(a) Species Richness — {res_label}",
        f"(b) Priority Index — {res_label}",
        f"(c) Priority Tier — {res_label}",
        f"(d) Flagship Species — {res_label}",
    ]

    composite_lbl_fs = 2.5 if is_500m else LABEL_FONTSIZE_SM

    for i, ax in enumerate(axes.flat):
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
        gdf[unocc_mask].plot(ax=ax, color=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                             linewidth=0.2, alpha=UNOCC_ALPHA, zorder=2)
        aoi_4326.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=1.2, zorder=5, linestyle="--")
        ax.set_title(PANEL_TITLES[i], loc="left", pad=4, fontsize=10, fontweight="bold")

    occ_gdf = gdf[occ_mask].copy()

    # Panel A: Richness
    ax0 = axes[0, 0]
    occ_gdf["_c"] = occ_gdf["species_richness"].apply(
        lambda v: mcolors.to_hex(cmap_S(norm_S(max(1, v)))))
    occ_gdf.plot(ax=ax0, color=occ_gdf["_c"].tolist(),
                 edgecolor=BORDER_COLOR, linewidth=0.2, alpha=OCC_ALPHA, zorder=3)
    draw_grid_labels(ax0, gdf, fontsize=composite_lbl_fs)
    cax0 = fig.add_axes([0.48, 0.565, 0.012, 0.36])
    ColorbarBase(cax0, cmap=cmap_S, norm=norm_S, orientation="vertical",
                 ticks=range(1, S_max + 1)).set_label("S", fontsize=8)

    # Panel B: Priority Index
    ax1 = axes[0, 1]
    occ_gdf["_c"] = occ_gdf["Priority_Index"].apply(
        lambda v: mcolors.to_hex(cmap_pri(norm_pri(v))) if pd.notna(v) else "#FFF")
    occ_gdf.plot(ax=ax1, color=occ_gdf["_c"].tolist(),
                 edgecolor=BORDER_COLOR, linewidth=0.2, alpha=OCC_ALPHA, zorder=3)
    draw_grid_labels(ax1, gdf, fontsize=composite_lbl_fs)
    cax1 = fig.add_axes([0.92, 0.565, 0.012, 0.36])
    ColorbarBase(cax1, cmap=cmap_pri, norm=norm_pri, orientation="vertical").set_label("PI", fontsize=8)

    # Panel C: Priority Tier
    ax2 = axes[1, 0]
    for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        subset = occ_gdf[occ_gdf["Priority_Tier"] == tier]
        if len(subset):
            subset.plot(ax=ax2, color=TIER_COLORS[tier], edgecolor=BORDER_COLOR,
                        linewidth=0.2, alpha=OCC_ALPHA, zorder=3)
    draw_grid_labels(ax2, gdf, fontsize=composite_lbl_fs)
    tier_handles = [mpatches.Patch(fc=TIER_COLORS[t], ec=BORDER_COLOR, alpha=OCC_ALPHA,
                    label=f"{t} ({tier_counts.get(t, 0)})") for t in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]]
    ax2.legend(handles=tier_handles, loc="upper left", fontsize=6, framealpha=0.85,
               edgecolor="#CCC", handlelength=1.0)

    # Panel D: Flagship
    ax3 = axes[1, 1]
    for count in sorted(occ_gdf["flagship_count"].unique()):
        subset = occ_gdf[occ_gdf["flagship_count"] == count]
        color = FLAGSHIP_COLORS.get(int(count), "#FFF")
        subset.plot(ax=ax3, color=color, edgecolor=BORDER_COLOR,
                    linewidth=0.2, alpha=OCC_ALPHA, zorder=3)
    draw_grid_labels(ax3, gdf, fontsize=composite_lbl_fs)

    fig.text(0.5, 0.005,
             f"BMP Conservation Priority Analysis — {res_label} square grid · "
             f"UCPS Landscape, West Java · Survey data 2009–2024 · "
             f"Basemap: Sentinel-2 (30 Sep 2024)",
             ha="center", fontsize=8, color="#444444", style="italic")

    save_fig(fig, f"{prefix}_E_composite")


# Generate maps for both resolutions
print("\n" + "=" * 70)
print("GENERATING BMP PRIORITY MAPS — 1 km GRID")
print("=" * 70)
make_priority_maps(grid_1km_4326, "1 km", "BMP_1km")

print("\n" + "=" * 70)
print("GENERATING BMP PRIORITY MAPS — 500 m GRID")
print("=" * 70)
make_priority_maps(grid_500m_4326, "500 m", "BMP_500m")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: REEPS DISTRIBUTION MAP (2025 — current state of knowledge)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING REEPS DISTRIBUTION MAP 2025")
print("=" * 70)

# Convert occurrences to 4326
occ_4326 = occ.to_crs("EPSG:4326")

# Filter to key REEPS species (>= 3 records for meaningful mapping)
sp_counts = occ_4326["Species"].value_counts()
key_species = sp_counts[sp_counts >= 3].index.tolist()
# Focus on scientifically named species (exclude local names)
key_species = [sp for sp in key_species if sp[0].isupper() and " " in sp]
print(f"  Key species for distribution map: {len(key_species)}")

# ── Fig: Multi-species distribution on 1km grid ─────────────────────
print("  Building REEPS distribution map ...")

fig = plt.figure(figsize=(9.0, 7.0), facecolor="white")
ax = fig.add_axes([0.02, 0.02, 0.72, 0.96])
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
ax.set_aspect("equal")
ax.set_facecolor("none")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for sp in ax.spines.values():
    sp.set_edgecolor("#888888")
    sp.set_linewidth(0.5)
ax.grid(True, linestyle=":", linewidth=0.2, color="#BBBBBB", alpha=0.4, zorder=1)

add_basemap(ax)

# Draw 1km grid as base
grid_1km_4326.plot(ax=ax, facecolor="none", edgecolor="#999999",
                   linewidth=0.3, alpha=0.5, zorder=2)
aoi_4326.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=AOI_LW,
                       zorder=5, linestyle="--")

# Draw grid labels
draw_grid_labels(ax, grid_1km_4326, fontsize=4.5)

# Plot species points
handles_sp = []
for sp_name in key_species:
    sp_data = occ_4326[occ_4326["Species"] == sp_name]
    color = SP_COL.get(sp_name, "#888888")
    abbr = ABBREV.get(sp_name, sp_name)

    ax.scatter(sp_data.geometry.x, sp_data.geometry.y,
               c=color, s=18, alpha=0.75, edgecolors="white",
               linewidths=0.3, zorder=8, label=abbr)

    handles_sp.append(Line2D([0], [0], marker="o", color="none",
                             markerfacecolor=color, markeredgecolor="white",
                             markeredgewidth=0.3, markersize=6,
                             label=f"{abbr} (n={len(sp_data)})"))

handles_sp.append(aoi_handle())
handles_sp.append(s2_patch())

ax.set_title("REEPS Species Distribution — Current Knowledge (2025)",
             loc="left", pad=6, fontweight="bold", fontsize=11)

# Legend on the right
ax.legend(handles=handles_sp, loc="upper left",
          bbox_to_anchor=(1.02, 1.0),
          fontsize=6.5, framealpha=0.90,
          edgecolor="#CCCCCC", handlelength=1.5, labelspacing=0.4,
          borderaxespad=0)

add_north_arrow(ax)
add_scalebar(ax)

add_subtitle(ax, f"Distribution of {len(key_species)} REEPS species across UCPS landscape · "
                 f"Total {len(occ_4326)} occurrence records from 2009–2024 surveys · "
                 f"Grid: 1 km BMP cells · Basemap: Sentinel-2 (30 Sep 2024)")

save_fig(fig, "REEPS_Distribution_2025")


# ── Fig: Per-species panel map (top 8 species) ──────────────────────
print("  Building per-species panels ...")
top_species = [sp for sp in key_species if sp in SP_COL][:8]
n_sp = len(top_species)
ncols = 4
nrows = (n_sp + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4.0), facecolor="white",
                          gridspec_kw={"hspace": 0.15, "wspace": 0.05})
if nrows == 1:
    axes = axes.reshape(1, -1)

for i, ax in enumerate(axes.flat):
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
    grid_1km_4326.plot(ax=ax, facecolor="none", edgecolor="#999999",
                       linewidth=0.15, alpha=0.4, zorder=2)
    aoi_4326.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=1.0, zorder=5, linestyle="--")

    # Draw grid labels on each panel
    draw_grid_labels(ax, grid_1km_4326, fontsize=3.0)

    if i < n_sp:
        sp_name = top_species[i]
        sp_data = occ_4326[occ_4326["Species"] == sp_name]
        color = SP_COL.get(sp_name, "#888888")
        abbr = ABBREV.get(sp_name, sp_name)

        # Highlight occupied grid cells
        joined = gpd.sjoin(sp_data, grid_1km_4326, how="inner", predicate="within")
        if len(joined) > 0:
            occ_cells = grid_1km_4326.loc[joined["index_right"].unique()]
            occ_cells.plot(ax=ax, facecolor=color, edgecolor=BORDER_COLOR,
                          linewidth=0.3, alpha=0.35, zorder=3)

        ax.scatter(sp_data.geometry.x, sp_data.geometry.y,
                   c=color, s=12, alpha=0.8, edgecolors="white",
                   linewidths=0.2, zorder=8)

        n_cells = len(joined["index_right"].unique()) if len(joined) > 0 else 0
        ax.set_title(f"{abbr} (n={len(sp_data)}, {n_cells} cells)",
                     loc="left", pad=3, fontsize=9, fontweight="bold")
    else:
        ax.set_visible(False)

fig.suptitle("REEPS Species Distribution by Species — UCPS Landscape (2025)",
             fontsize=13, fontweight="bold", y=0.98)
fig.text(0.5, 0.01,
         "Individual species occurrence records and occupied 1 km grid cells · "
         "Survey data 2009–2024 · Basemap: Sentinel-2 (30 Sep 2024)",
         ha="center", fontsize=8, color="#444444", style="italic")

save_fig(fig, "REEPS_Distribution_2025_PerSpecies")

print("\n" + "=" * 70)
print(f"ALL DONE! Output: {OUT_DIR}")
print("Files generated:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
print("=" * 70)
