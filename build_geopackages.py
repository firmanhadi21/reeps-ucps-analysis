#!/usr/bin/env python3
"""Build fresh GeoPackages from the authoritative CSV analysis outputs.

Why this exists
---------------
The original packages, REEPS_Master_Database.gpkg and REEPS_GridAnalyses.gpkg, hold
the *unfiltered* database: 596 occurrence records across 26 taxa in 41 occupied cells
on a 150-cell grid. The analysis reported in the manuscript uses the filtered set:
493 records across the 11 REEPS species in 38 occupied cells on a 149-cell grid, with
one transcribed longitude corrected. The packages were never refreshed after that
correction, so every figure built from them carried pre-correction numbers.

This script writes REEPS_Master_Database_v2.gpkg and REEPS_GridAnalyses_v2.gpkg from
the CSVs that the analysis actually produces. Layer names and column names match the
originals exactly, so a consuming script needs only to change the filename.

The originals are left untouched. REEPS_Master_Database.gpkg in particular is the
marker file that every script's _reeps_base() walks the tree to find, so removing or
renaming it would break path resolution project-wide.

Columns deliberately dropped
----------------------------
Eco__Score, Trend_Direction, Trend_Slope and Diversity_Trend belonged to the
Ecological Importance Score and the temporal-trend analyses, both removed from the
study in revision. No consuming script references them.
"""

import os as _os
from pathlib import Path as _Path


def _reeps_base() -> _Path:
    env = _os.environ.get("REEPS_BASE")
    if env:
        return _Path(env).expanduser().resolve()
    here = _Path(__file__).resolve()
    for cand in (here.parent, *here.parents):
        if (cand / "REEPS_Master_Database.gpkg").exists():
            return cand
    return here.parent


REEPS_BASE = _reeps_base()

import warnings
warnings.filterwarnings("ignore")

import os

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import Point, Polygon

os.chdir(str(REEPS_BASE))

MASTER_OUT = "REEPS_Master_Database_v2.gpkg"
GRID_OUT = "REEPS_GridAnalyses_v2.gpkg"


def hex_poly(cell: str) -> Polygon:
    return Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])


def hex_frame(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(df.copy(),
                            geometry=[hex_poly(c) for c in df["h3_index"]],
                            crs="EPSG:4326")


def write(gdf: gpd.GeoDataFrame, path: str, layer: str, first: bool) -> None:
    gdf.to_file(path, layer=layer, driver="GPKG",
                mode="w" if first else "a")
    print(f"    [{layer}]  {len(gdf)} rows, {len(gdf.columns) - 1} fields")


def build_master() -> None:
    print(f"{MASTER_OUT}")
    if os.path.exists(MASTER_OUT):
        os.remove(MASTER_OUT)

    occ = pd.read_csv("reeps_h3.csv")
    gdf = gpd.GeoDataFrame(
        occ.copy(),
        geometry=[Point(x, y) for x, y in zip(occ["Longitude"], occ["Latitude"])],
        crs="EPSG:4326")
    write(gdf, MASTER_OUT, "reeps_occurrences", first=True)

    h3a = pd.read_csv("h3_analysis.csv")
    cells = h3a.rename(columns={
        "lat": "Lat", "lon": "Lon",
        "total_records": "Total_Records",
        "species_richness": "Species_Richness",
        "species_list": "Species_List",
        "first_year": "First_Year", "last_year": "Last_Year",
        "years_w_data": "Years_w__Data",
        **{f"records_{y}": f"Records_{y}"
           for y in (2009, 2012, 2014, 2017, 2018, 2020, 2022, 2024, 2025, 2026)},
    })
    keep = ["h3_index", "Lat", "Lon", "Total_Records", "Species_Richness",
            "Species_List", "First_Year", "Last_Year", "Years_w__Data"] + [
        f"Records_{y}" for y in
        (2009, 2012, 2014, 2017, 2018, 2020, 2022, 2024, 2025, 2026)]
    write(hex_frame(cells[keep]), MASTER_OUT, "reeps_h3_cells", first=False)

    aoi = gpd.read_file("aoi.gpkg").to_crs("EPSG:4326")
    write(aoi, MASTER_OUT, "aoi_boundary", first=False)


def build_grid() -> None:
    print(f"\n{GRID_OUT}")
    if os.path.exists(GRID_OUT):
        os.remove(GRID_OUT)

    conn = pd.read_csv("h3_connectivity_full.csv")
    allc = conn.rename(columns={
        "cell_type": "Cell_Type", "occupied": "Occupied",
        "patch_id": "Patch_ID", "occ_nbrs": "Occ__Nbrs",
        "k2_occ_nbrs": "K2_Occ__Nbrs", "sp_reachable_k2": "Sp__Reachable_K2",
        "structural_class": "Structural_Class", "keystone": "Keystone",
    })[["h3_index", "lat", "lon", "Cell_Type", "Structural_Class", "Occupied",
        "Keystone", "Patch_ID", "Occ__Nbrs", "K2_Occ__Nbrs", "Sp__Reachable_K2"]]
    write(hex_frame(allc), GRID_OUT, "h3_all_cells", first=True)

    h3a = pd.read_csv("h3_analysis.csv")
    rich = h3a.rename(columns={
        "total_records": "Total_Records", "species_richness": "Species_Richness",
        "species_list": "Species_List", "first_year": "First_Year",
        "last_year": "Last_Year", "years_w_data": "Years_w__Data",
        **{f"records_{y}": f"Records_{y}"
           for y in (2009, 2012, 2014, 2017, 2018, 2020, 2022, 2024, 2025, 2026)},
    })
    keep = ["h3_index", "lat", "lon", "Total_Records", "Species_Richness",
            "Species_List", "First_Year", "Last_Year", "Years_w__Data"] + [
        f"Records_{y}" for y in
        (2009, 2012, 2014, 2017, 2018, 2020, 2022, 2024, 2025, 2026)]
    write(hex_frame(rich[keep]), GRID_OUT, "h3_richness_summary", first=False)

    div = pd.read_csv("h3_diversity.csv").rename(columns={
        "richness_S": "Richness__S", "records_N": "Records__N",
        "shannon_H": "Shannon__H", "simpson_D": "Simpson__D",
        "pielou_J": "Pielou__J", "berger_parker_BP": "Berger_Parker__BP",
        "dominant_species": "Dominant_Species", "dominant_common": "Common_Name",
    })[["h3_index", "lat", "lon", "Richness__S", "Records__N", "Shannon__H",
        "Simpson__D", "Pielou__J", "Berger_Parker__BP", "Dominant_Species",
        "Common_Name"]]
    write(hex_frame(div), GRID_OUT, "h3_diversity", first=False)

    pri = pd.read_csv("priority_index.csv").rename(
        columns={"Priority_Tier": "Tier"})
    keep = ["Rank", "h3_index", "Tier", "Richness", "Diversity", "Connectivity",
            "Co_occurrence", "Threatened", "Permeability", "Priority_Index",
            "lat", "lon"]
    write(hex_frame(pri[keep]), GRID_OUT, "h3_priority", first=False)


def verify() -> None:
    print("\nverification")
    checks = [
        (MASTER_OUT, "reeps_occurrences", 493),
        (MASTER_OUT, "reeps_h3_cells", 149),
        (GRID_OUT, "h3_all_cells", 149),
        (GRID_OUT, "h3_richness_summary", 149),
        (GRID_OUT, "h3_diversity", 38),
        (GRID_OUT, "h3_priority", 38),
    ]
    ok = True
    for path, layer, expect in checks:
        g = gpd.read_file(path, layer=layer)
        good = len(g) == expect
        ok &= good
        print(f"  {layer:22s} {len(g):4d} rows (expected {expect})  "
              f"{'ok' if good else 'MISMATCH'}")

    occ = gpd.read_file(MASTER_OUT, layer="reeps_occurrences")
    print(f"  species in occurrences : {occ['Species'].nunique()} (expected 11)")
    print(f"  occupied cells         : {occ['h3_index'].nunique()} (expected 38)")
    print(f"  longitude range        : {occ.geometry.x.min():.4f} .. "
          f"{occ.geometry.x.max():.4f}")
    rich = gpd.read_file(GRID_OUT, layer="h3_richness_summary")
    print(f"  cells with richness>0  : {(rich['Species_Richness'] > 0).sum()} "
          f"(expected 38)")
    print(f"  max richness           : {rich['Species_Richness'].max()} "
          f"(expected 10)")
    print("\n" + ("ALL CHECKS PASSED" if ok else "CHECKS FAILED"))


if __name__ == "__main__":
    build_master()
    build_grid()
    verify()
