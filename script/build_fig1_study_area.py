#!/usr/bin/env python3
"""
Figure 1: Study Area Map.

Rebuilt in response to Reviewer 2's comment 6, which asked for a higher-resolution
basemap, better contrast, and geographic rather than UTM coordinates:

  * basemap is now the local Sentinel-2 S2DR3 super-resolved scene (1 m native,
    resampled to 3000 px across the AOI) instead of zoom-13 web tiles, which were
    too coarse to identify locations;
  * a 2-98 percentile contrast stretch is applied per band, since the raw scene is
    dominated by dark forest and reads as a flat green mass;
  * axes are WGS84 with degree/minute/second tick labels;
  * the regional inset sits outside the map frame so it no longer covers data.

Cell counts in the legend are read from the analysis outputs, so they track the
corrected 38-of-149 tessellation rather than being hardcoded.
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

import os

import geopandas as gpd
import h3
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.ticker import FuncFormatter
from rasterio.transform import array_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import Polygon, box

os.chdir(str(REEPS_BASE))

S2_TIF = "Sentinel2_S2DR3_30Sep24_RGB.tif"
# 5000 px across the ~0.196 deg AOI is ~4.3 m/px, against ~19 m/px for the zoom-13
# web tiles this figure previously used. Figure 1 is a full-page single map, so it
# carries a finer basemap than the multi-panel figures (3000 px).
TARGET_W = 5000
DST_CRS = "EPSG:4326"

# Inset extent: Sumatra, Kalimantan, Java, Bali
INSET_LON = (95, 120)
INSET_LAT = (-10, 7)


def dms(value: float, axis: str) -> str:
    """Format a signed decimal degree as degrees/minutes/seconds with a hemisphere."""
    hemi = ("N" if value >= 0 else "S") if axis == "lat" else ("E" if value >= 0 else "W")
    v = abs(value)
    d = int(v)
    m_float = (v - d) * 60
    m = int(m_float)
    s = (m_float - m) * 60
    if round(s) == 60:          # carry, so 6°59'60" prints as 7°00'00"
        s, m = 0.0, m + 1
    if m == 60:
        m, d = 0, d + 1
    return f"{d}°{m:02d}'{round(s):02d}\"{hemi}"


def stretch(band: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Percentile contrast stretch to [0, 1].

    Reviewer 2 noted the true-colour composite has very limited contrast between
    land-cover features. The scene is mostly closed-canopy forest, so the raw
    digital numbers occupy a narrow part of the range; clipping to the 2nd and
    98th percentiles of the valid pixels spreads them across the full range.
    """
    valid = band[band > 0]
    if valid.size == 0:
        return band.astype(np.float32)
    lo, hi = np.percentile(valid, [lo_pct, hi_pct])
    if hi <= lo:
        return band.astype(np.float32) / 255.0
    return np.clip((band.astype(np.float32) - lo) / (hi - lo), 0, 1)


def load_basemap():
    """Warp the Sentinel-2 scene to WGS84, resample, and stretch each band."""
    print("Loading Sentinel-2 S2DR3 basemap (warping to WGS84) …")
    with rasterio.open(S2_TIF) as src:
        transform_native, width_native, height_native = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds
        )
        scale = TARGET_W / width_native
        dst_w = TARGET_W
        dst_h = max(1, int(height_native * scale))

        b = array_bounds(height_native, width_native, transform_native)
        dst_transform = transform_from_bounds(b[0], b[1], b[2], b[3], dst_w, dst_h)

        rgb = np.zeros((3, dst_h, dst_w), dtype=np.uint8)
        for i in range(1, 4):
            reproject(
                source=rasterio.band(src, i),
                destination=rgb[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=DST_CRS,
                resampling=Resampling.bilinear,
            )

    img = np.zeros((dst_h, dst_w, 3), dtype=np.float32)
    for i in range(3):
        img[:, :, i] = stretch(rgb[i])

    extent = (b[0], b[2], b[1], b[3])   # left, right, bottom, top
    print(f"  {dst_w}×{dst_h} px, extent "
          f"[{extent[0]:.4f}, {extent[1]:.4f}] × [{extent[2]:.4f}, {extent[3]:.4f}]")
    return img, extent


def _regional_inset(ax, aoi_bounds) -> None:
    """Locator map of the western Indonesian archipelago with the AOI marked.

    OpenStreetMap's tile servers reject contextily's default user agent, so we try
    a short list of providers and fall back to a plain extent box rather than
    letting the figure carry "Access blocked" tiles.
    """
    import contextily as ctx

    bnds = (
        gpd.GeoSeries([box(INSET_LON[0], INSET_LAT[0], INSET_LON[1], INSET_LAT[1])],
                      crs="EPSG:4326").to_crs(epsg=3857).total_bounds
    )
    ax.set_xlim(bnds[0], bnds[2])
    ax.set_ylim(bnds[1], bnds[3])

    providers = [
        ("CartoDB Positron", ctx.providers.CartoDB.Positron),
        ("CartoDB Voyager", ctx.providers.CartoDB.Voyager),
        ("Esri WorldGrayCanvas", ctx.providers.Esri.WorldGrayCanvas),
    ]
    for name, prov in providers:
        try:
            ctx.add_basemap(ax, source=prov, zoom=5, attribution=False)
            print(f"  inset basemap: {name}")
            break
        except Exception as exc:
            print(f"  inset basemap {name} failed ({type(exc).__name__}); trying next")
    else:
        print("  no inset basemap available; drawing a plain locator box")
        ax.set_facecolor("#EEF3F7")

    marker = gpd.GeoSeries(
        [box(*aoi_bounds)], crs="EPSG:4326").to_crs(epsg=3857)
    # At this scale the AOI is sub-pixel, so mark it with a visible symbol too.
    marker.boundary.plot(ax=ax, color="red", linewidth=1.6)
    c = marker.geometry.iloc[0].centroid
    ax.plot(c.x, c.y, marker="o", markersize=7, markerfacecolor="none",
            markeredgecolor="red", markeredgewidth=1.6)
    ax.annotate("UCPS", xy=(c.x, c.y), xytext=(14, -20),
                textcoords="offset points", fontsize=8, fontweight="bold",
                color="red",
                arrowprops=dict(arrowstyle="-", color="red", lw=0.9))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


def h3_to_polygon(cell: str) -> Polygon:
    bnd = h3.cell_to_boundary(cell)
    return Polygon([(lon, lat) for lat, lon in bnd])


def load_cells():
    """H3 cells and occupancy, taken from the analysis output so counts stay current."""
    df = pd.read_csv("h3_analysis.csv")
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[h3_to_polygon(c) for c in df["h3_index"]],
        crs="EPSG:4326",
    )
    occ = gdf[gdf["total_records"] > 0]
    print(f"  Occupied cells: {len(occ)} of {len(gdf)}")
    return gdf, occ


def main():
    aoi = gpd.read_file("aoi.gpkg").to_crs(DST_CRS)
    gdf_all, gdf_occ = load_cells()
    img, extent = load_basemap()

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    b = aoi.total_bounds
    pad_x = (b[2] - b[0]) * 0.02
    pad_y = (b[3] - b[1]) * 0.02
    xmin, xmax = b[0] - pad_x, b[2] + pad_x
    ymin, ymax = b[1] - pad_y, b[3] + pad_y

    fig = plt.figure(figsize=(12, 9.2), facecolor="white")
    ax = fig.add_axes([0.08, 0.07, 0.90, 0.785])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect(1.0 / np.cos(np.radians(abs((ymin + ymax) / 2))))

    ax.imshow(img, extent=extent, origin="upper", interpolation="bilinear", zorder=0)
    gdf_all.plot(ax=ax, facecolor="white", alpha=0.10, edgecolor="white",
                 linewidth=0.4, zorder=2)
    gdf_occ.plot(ax=ax, facecolor="#3388ff", alpha=0.40, edgecolor="#1a5cb5",
                 linewidth=1.0, zorder=3)
    aoi.boundary.plot(ax=ax, linewidth=2.2, color="red", linestyle="--", zorder=10)

    # Geographic coordinates in degrees / minutes / seconds (Reviewer 2, comment 6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: dms(v, "lon")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: dms(v, "lat")))
    ax.set_xticks(np.linspace(xmin, xmax, 6))
    ax.set_yticks(np.linspace(ymin, ymax, 5))
    ax.tick_params(labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, color="white")
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)

    # North arrow
    ax.annotate("", xy=(0.965, 0.955), xytext=(0.965, 0.895),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6))
    ax.text(0.965, 0.960, "N", transform=ax.transAxes, fontsize=11,
            fontweight="bold", color="white", ha="center", va="bottom")

    # Scale bar, 2 km, in degrees of longitude at this latitude
    lat_mid = (ymin + ymax) / 2
    deg_per_km = 1.0 / (111.32 * np.cos(np.radians(abs(lat_mid))))
    blen = 2 * deg_per_km
    bx = xmin + (xmax - xmin) * 0.04
    by = ymin + (ymax - ymin) * 0.06
    ax.plot([bx, bx + blen], [by, by], "-", color="white", lw=3,
            solid_capstyle="butt", zorder=11)
    for xe in (bx, bx + blen):
        ax.plot([xe, xe], [by, by + (ymax - ymin) * 0.012], "-", color="white",
                lw=3, zorder=11)
    ax.text(bx + blen / 2, by + (ymax - ymin) * 0.018, "2 km", ha="center",
            fontsize=8.5, color="white", fontweight="bold", zorder=11)

    ax.text(0.99, 0.015,
            "Basemap: Sentinel-2 (S2DR3 super-resolved, 30 Sep 2024) · "
            "CRS: WGS 84 (EPSG:4326)",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75,
                      edgecolor="none"), zorder=11)

    legend_elements = [
        mpatches.Patch(facecolor="#3388ff", alpha=0.40, edgecolor="#1a5cb5",
                       linewidth=1.0,
                       label=f"H3 cells with REEPS records (n = {len(gdf_occ)})"),
        mpatches.Patch(facecolor="white", alpha=0.30, edgecolor="white",
                       linewidth=0.4,
                       label=f"All H3 cells in AOI (n = {len(gdf_all)})"),
        mpatches.Patch(facecolor="none", edgecolor="red", linewidth=2,
                       linestyle="--", label="AOI boundary"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5,
              framealpha=0.9, bbox_to_anchor=(1.0, 0.055))

    ax.text(0.012, 0.982, "(b)  Study area",
            transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            color="black", va="top", ha="left", zorder=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                      edgecolor="none"))

    # ── Regional inset, in its own band above the map so it covers no data ─────
    ax_in = fig.add_axes([0.08, 0.875, 0.30, 0.105])
    _regional_inset(ax_in, b)
    ax_in.set_title("(a)  Regional context", fontsize=9.5, fontweight="bold",
                    loc="left", pad=3)
    fig.text(0.40, 0.925,
             "Upper Cisokan Pumped Storage (UCPS), West Java, Indonesia",
             fontsize=12, fontweight="bold", va="center")

    out = "figures/fig1_study_area.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
