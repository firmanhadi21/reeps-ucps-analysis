"""
basemap_utils.py
─────────────────
Shared utilities for all REEPS static figure scripts.
Loads the Sentinel-2 true-colour basemap once and exposes
common figure-setup helpers so every map uses the same style.
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
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.colorbar import ColorbarBase
from shapely.geometry import Polygon
from pathlib import Path
import h3
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import array_bounds, from_bounds as transform_from_bounds

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = REEPS_BASE
BIO  = REEPS_BASE
S2_TIF = BIO / "Sentinel2_S2DR3_30Sep24_RGB.tif"

# ── Output root ────────────────────────────────────────────────────────────────
STATIC_OUT = BIO / "static_figures"
STATIC_OUT.mkdir(exist_ok=True)

# ── Species palette (shared across all maps) ──────────────────────────────────
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
    "Arctictis binturong":     "#4E342E",
    "Prionodon linsang":       "#37474F",
    "Paguma larvata":          "#78909C",
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
    "Arctictis binturong":     "Binturong",
    "Prionodon linsang":       "Banded Linsang",
    "Paguma larvata":          "Masked Palm Civet",
}

# ── Map style constants ────────────────────────────────────────────────────────
UNOCC_COLOR  = "#E8E8E8"
UNOCC_ALPHA  = 0.22
OCC_ALPHA    = 0.65
BORDER_COLOR = "#555555"
BORDER_LW    = 0.35
AOI_COLOR    = "#E65100"
AOI_LW       = 1.8
FIG_W, FIG_H = 7.0, 5.4   # inches – A4 column-width compatible

# ── Publication rcParams ───────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size":       9,
    "axes.titlesize":  11,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.linewidth":  0.6,
})


# ══════════════════════════════════════════════════════════════════════════════
# Sentinel-2 basemap — loaded once at module import
# ══════════════════════════════════════════════════════════════════════════════
# Width, in pixels, that the full S2DR3 scene is resampled to at import.
# The source is 21620 px at 1 m; at 3000 px the effective ground sample was
# ~7.2 m, which is what made the map panels look soft. 8000 px gives ~2.7 m.
# Override with REEPS_BASEMAP_W (e.g. 3000 for a fast low-memory preview).
_TARGET_W = int(_os.environ.get("REEPS_BASEMAP_W", 8000))

print("basemap_utils: loading Sentinel-2 basemap …")
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

# Public: RGBA array and matplotlib extent tuple.
# uint8 rather than float32 — imshow accepts it directly and it costs a quarter
# of the memory, which matters once _TARGET_W is in the thousands.
S2_RGBA = np.empty((_dst_height, _dst_width, 4), dtype=np.uint8)
S2_RGBA[:, :, 0] = _rgb[0]
S2_RGBA[:, :, 1] = _rgb[1]
S2_RGBA[:, :, 2] = _rgb[2]
S2_RGBA[:, :, 3] = 255
del _rgb
# matplotlib imshow extent = (left, right, bottom, top)
S2_EXTENT = (_lc_bounds[0], _lc_bounds[2], _lc_bounds[1], _lc_bounds[3])
print(f"  Basemap ready: {_dst_width}×{_dst_height} px  "
      f"({S2_RGBA.nbytes / 1e6:.0f} MB, "
      f"~{111320 * (S2_EXTENT[1] - S2_EXTENT[0]) / _dst_width:.1f} m/px)  "
      f"[{S2_EXTENT[0]:.4f}–{S2_EXTENT[1]:.4f}°E, "
      f"{S2_EXTENT[2]:.4f}–{S2_EXTENT[3]:.4f}°N]")


# ══════════════════════════════════════════════════════════════════════════════
# H3 helpers
# ══════════════════════════════════════════════════════════════════════════════
def h3_to_polygon(cell):
    bnd = h3.cell_to_boundary(cell)
    return Polygon([(lon, lat) for lat, lon in bnd])


# ══════════════════════════════════════════════════════════════════════════════
# AOI + map extent
# ══════════════════════════════════════════════════════════════════════════════
aoi = gpd.read_file(BIO / "aoi.gpkg").to_crs(4326)
_ab = aoi.total_bounds
_px = (_ab[2] - _ab[0]) * 0.10
_py = (_ab[3] - _ab[1]) * 0.10
XMIN, YMIN = _ab[0] - _px, _ab[1] - _py
XMAX, YMAX = _ab[2] + _px, _ab[3] + _py


# ══════════════════════════════════════════════════════════════════════════════
# Shared drawing functions
# ══════════════════════════════════════════════════════════════════════════════
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


def draw_hex_base(ax, gdf_unocc):
    """Light ghost borders for unoccupied cells."""
    gdf_unocc.plot(ax=ax, color=UNOCC_COLOR, edgecolor=BORDER_COLOR,
                   linewidth=BORDER_LW, alpha=UNOCC_ALPHA, zorder=2)
    aoi.boundary.plot(ax=ax, color=AOI_COLOR, linewidth=AOI_LW,
                      zorder=5, linestyle="--")


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


def add_subtitle(ax, text):
    ax.text(0.5, -0.01, text, transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color="#444444",
            style="italic", wrap=True)


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


def save_fig(fig, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"  ✓  {out_dir.name}/{stem}.png / .pdf")
    plt.close(fig)


def std_colorbar(fig, cax_rect, cmap, norm, label):
    cax = fig.add_axes(cax_rect)
    cb  = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label(label, fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    return cb
