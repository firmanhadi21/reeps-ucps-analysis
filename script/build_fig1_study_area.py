#!/usr/bin/env python3
"""
Figure 1: Study Area Map
- Inset: OSM showing Sumatera, Kalimantan, Java, Bali with AOI box
- Main: Sentinel-2 RGB basemap with transparent H3 grid overlay showing recorded species
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


import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from shapely.geometry import box
import contextily as ctx
import numpy as np
import os
from xyzservices import TileProvider
from matplotlib.ticker import FuncFormatter

# Set working directory
os.chdir(str(REEPS_BASE))

# AOI bounds (lon/lat)
LON_MIN, LON_MAX = 107.15, 107.30
LAT_MIN, LAT_MAX = -6.95, -6.85
UTM_CRS = "EPSG:32748"


def main():
    # Load data
    print("Loading data...")
    aoi = gpd.read_file("aoi.gpkg")
    h3_cells = gpd.read_file("REEPS_GridAnalyses.gpkg", layer="h3_all_cells")
    reeps_occ = gpd.read_file("REEPS_Master_Database.gpkg", layer="reeps_occurrences")

    # Get occupied cells (column uses 1/0 integers, or "Yes"/"No" strings)
    if h3_cells["Occupied"].dtype == object:
        occupied_cells = h3_cells[h3_cells["Occupied"] == "Yes"]
    else:
        occupied_cells = h3_cells[h3_cells["Occupied"] == 1]
    print(f"  Occupied cells: {len(occupied_cells)} of {len(h3_cells)}")

    # Project to UTM Zone 48S for correct axis labels
    aoi_utm = aoi.to_crs(UTM_CRS)
    h3_utm = h3_cells.to_crs(UTM_CRS)
    occ_utm = occupied_cells.to_crs(UTM_CRS)

    # AOI bounds in UTM
    aoi_bounds = aoi_utm.total_bounds  # minx, miny, maxx, maxy

    # Create figure with main map and inset
    fig = plt.figure(figsize=(12, 10))

    # Main map axes
    ax_main = fig.add_axes([0.1, 0.1, 0.85, 0.85])  # [left, bottom, width, height]

    # Set extent for main map (UTM Zone 48S)
    ax_main.set_xlim(aoi_bounds[0], aoi_bounds[2])
    ax_main.set_ylim(aoi_bounds[1], aoi_bounds[3])

    # Sentinel-2 RGB basemap (s2cloudless)
    s2_tiles = TileProvider(
        name="Sentinel-2 cloudless",
        url="https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg",
        attribution="Sentinel-2 cloudless (EOX)",
    )
    ctx.add_basemap(ax_main, source=s2_tiles, zoom=13, crs=UTM_CRS)

    # Plot H3 grid with transparency - all cells first (underneath)
    h3_utm.plot(ax=ax_main, facecolor="lightgray", alpha=0.15, edgecolor="gray",
                linewidth=0.4, zorder=2)

    # Occupied cells with REEPS records - clearly visible blue fill
    occ_utm.plot(ax=ax_main, facecolor="#3388ff", alpha=0.45, edgecolor="#1a5cb5",
                 linewidth=1.0, zorder=3)

    # Plot AOI boundary ON TOP with dashed red line (clearly visible)
    if not aoi_utm.empty:
        aoi_utm.boundary.plot(ax=ax_main, linewidth=2.5, color="red",
                              linestyle="--", zorder=10)

    # Add gridlines
    ax_main.grid(True, alpha=0.2, linestyle="--")

    # Force full numeric tick labels (no abbreviation)
    ax_main.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):d}"))
    ax_main.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{int(y):d}"))

    # Labels
    ax_main.set_xlabel("Easting (m)", fontsize=12)
    ax_main.set_ylabel("Northing (m)", fontsize=12)
    ax_main.set_title(
        "Study Area: Upper Cisokan Pumped Storage (UCPS), West Java, Indonesia",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    # North arrow
    ax_main.annotate(
        "N",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        fontsize=14,
        ha="center",
        fontweight="bold",
    )
    ax_main.annotate(
        "↑", xy=(0.95, 0.93), xycoords="axes fraction", fontsize=18, ha="center"
    )

    # Scale bar (approximate) - 2 km
    scale_length = 2000  # meters
    x0 = aoi_bounds[0] + (aoi_bounds[2] - aoi_bounds[0]) * 0.05
    y0 = aoi_bounds[1] + (aoi_bounds[3] - aoi_bounds[1]) * 0.07
    ax_main.plot([x0, x0 + scale_length], [y0, y0], "k-", linewidth=2)
    ax_main.plot([x0, x0], [y0, y0 - 200], "k-", linewidth=2)
    ax_main.plot(
        [x0 + scale_length, x0 + scale_length], [y0, y0 - 200], "k-", linewidth=2
    )
    ax_main.text(x0 + scale_length / 2, y0 - 500, "2 km", ha="center", fontsize=9)

    # CRS / Datum note
    ax_main.text(
        0.99,
        0.02,
        "CRS: WGS 84 / UTM zone 48S (EPSG:32748)",
        transform=ax_main.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )

    # Legend for main map (colors match actual plot)
    legend_elements = [
        mpatches.Patch(
            facecolor="#3388ff",
            alpha=0.45,
            edgecolor="#1a5cb5",
            linewidth=1.0,
            label=f"H3 cells with REEPS records (n={len(occ_utm)})",
        ),
        mpatches.Patch(
            facecolor="lightgray",
            alpha=0.15,
            edgecolor="gray",
            linewidth=0.4,
            label=f"All H3 cells (n={len(h3_utm)})",
        ),
        mpatches.Patch(
            facecolor="none", edgecolor="red", linewidth=2,
            linestyle="--", label="AOI boundary"
        ),
    ]
    ax_main.legend(
        handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9
    )

    # ============ INSET MAP ============
    # Create inset showing regional context (Java, Sumatera, Kalimantan, Bali)
    ax_inset = fig.add_axes([0.02, 0.55, 0.25, 0.30])  # [left, bottom, width, height]

    # Extent to cover Sumatera, Kalimantan, Java, Bali (lon/lat)
    inset_lon_min, inset_lon_max = 95, 120
    inset_lat_min, inset_lat_max = -10, 7

    inset_bounds_3857 = (
        gpd.GeoSeries(
            [box(inset_lon_min, inset_lat_min, inset_lon_max, inset_lat_max)],
            crs="EPSG:4326",
        )
        .to_crs(epsg=3857)
        .total_bounds
    )

    ax_inset.set_xlim(inset_bounds_3857[0], inset_bounds_3857[2])
    ax_inset.set_ylim(inset_bounds_3857[1], inset_bounds_3857[3])

    # Add OSM background for inset
    ctx.add_basemap(ax_inset, source=ctx.providers.OpenStreetMap.Mapnik, zoom=5)

    # Add box for the AOI extent
    aoi_box = gpd.GeoSeries(
        [box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    aoi_box.boundary.plot(ax=ax_inset, color="red", linewidth=2)

    ax_inset.axis("off")
    ax_inset.set_title(
        "(a) Regional context", fontsize=10, fontweight="bold", loc="left"
    )

    # Add label for main map
    ax_main.text(
        0.02,
        0.98,
        "(b) Study area",
        transform=ax_main.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )

    plt.tight_layout()

    # Save
    output_path = "figures/fig1_study_area_new.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(
        output_path.replace(".pdf", ".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved: {output_path}")

    # Also save as fig1_study_area for direct use
    import shutil

    shutil.copy(output_path, "figures/fig1_study_area.pdf")
    shutil.copy(output_path.replace(".pdf", ".png"), "figures/fig1_study_area.png")
    print("Updated fig1_study_area in figures/")

    plt.close()


if __name__ == "__main__":
    main()
