#!/usr/bin/env python3
"""
prepare_data_release.py
───────────────────────
Builds the public data package for the code release.

Every analysis in the paper is conducted at the level of the H3 resolution-8
cell, so releasing occurrences aggregated to those cells reproduces all published
results exactly while withholding point localities. This matters because 214 of
the 493 analysed records (43.4%) are of Critically Endangered species subject to
illegal collection -- Javan Leopard, Javan Slow Loris and Sunda Pangolin.

Writes release/data/:
    reeps_h3_aggregated.csv   species x cell x period counts, no coordinates
    h3_cell_centroids.csv     cell centroids (cell resolution only, ~0.74 km2)
    <derived tables>          the analysis outputs, which are already cell-level

Use --full to include point coordinates instead. That is a deliberate choice and
requires the data owners' agreement; the script refuses unless the flag is given.
"""

import os
import shutil
import sys
from pathlib import Path

import h3
import pandas as pd


def _reeps_base() -> Path:
    env = os.environ.get("REEPS_BASE")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for cand in (here.parent, *here.parents):
        if (cand / "REEPS_Master_Database.gpkg").exists():
            return cand
    return here.parent


BASE = _reeps_base()
OUT = BASE / "release" / "data"

# derived tables that are already cell-level and carry no point localities
DERIVED = [
    "h3_analysis.csv", "h3_diversity.csv", "priority_index.csv",
    "h3_connectivity_full.csv", "gap_analysis.csv", "stepping_stones.csv",
    "cooccurrence_pairs.csv", "cooccurrence_species.csv",
    "cooccurrence_significance_per_cell.csv", "presence_absence.csv",
    "turnover_cell_period.csv", "survey_coverage.csv",
    "cpi_correlation_matrix.csv", "pca_cpi_weights.csv",
    "h3_resolution_comparison.csv", "chao1_comparison.csv",
    "chao1_per_period.csv", "effort_per_period.csv",
    # Land-cover derived, no species content. The aggregated one is required:
    # compute_cpi_pca6.py reads it for the habitat-permeability component, so
    # without it the CPI stage cannot run at all.
    "resistance_res9_aggregated_to_res8.csv", "resistance_res9_results.csv",
    "resistance_corridor_results.csv",
    "morans_i_results.csv", "lisa_results.csv",
]

CR = {"Panthera pardus melas", "Nycticebus javanicus", "Manis javanica"}


def main() -> None:
    full = "--full" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)

    occ = pd.read_csv(BASE / "reeps_h3.csv")
    n_cr = int(occ["Species"].isin(CR).sum())
    print(f"{len(occ)} analysed records; {n_cr} ({100*n_cr/len(occ):.1f}%) "
          f"are Critically Endangered species")

    if full:
        print("\n--full given: including point coordinates.")
        occ.to_csv(OUT / "reeps_occurrences_full.csv", index=False)
        print(f"  wrote reeps_occurrences_full.csv ({len(occ)} rows) "
              f"WITH coordinates")
    else:
        # dropna=False is essential: 16 of the 493 records carry no survey year,
        # and pandas drops NaN group keys by default. Without it the package loses
        # those records and Table 1 no longer reproduces — the aggregate came to
        # 477 records, with seven species undercounted and two of them short a
        # cell as well.
        agg = (occ.groupby(["h3_index", "Species", "Year"], dropna=False)
               .size().reset_index(name="records"))
        agg["Year"] = agg["Year"].astype("Int64")
        agg.to_csv(OUT / "reeps_h3_aggregated.csv", index=False)
        print(f"\n  wrote reeps_h3_aggregated.csv "
              f"({len(agg)} species-cell-period rows, no coordinates)")

        # The package is worthless to a reviewer if it does not reproduce the
        # paper, so refuse to emit one that does not.
        rebuilt = agg.groupby("Species").agg(
            records=("records", "sum"), cells=("h3_index", "nunique"))
        source = occ.groupby("Species").agg(
            records=("Species", "size"), cells=("h3_index", "nunique"))
        mismatch = [(s, tuple(rebuilt.loc[s]), tuple(source.loc[s]))
                    for s in source.index
                    if tuple(rebuilt.loc[s]) != tuple(source.loc[s])]
        assert not mismatch, f"aggregate does not reproduce Table 1: {mismatch}"
        assert int(rebuilt["records"].sum()) == len(occ), (
            f"aggregate holds {int(rebuilt['records'].sum())} records, "
            f"source has {len(occ)}")
        print(f"  checked: rebuilds Table 1 exactly "
              f"({int(rebuilt['records'].sum())} records, "
              f"{len(rebuilt)} species)")

        # The per-record table with the sensitive columns removed. This is what
        # makes the package runnable: four of the analysis scripts read
        # reeps_h3.csv and expect one row per record, so shipping only the
        # aggregate leaves a reviewer unable to execute the pipeline at all.
        # Dropping Latitude, Longitude and Location gives identical protection —
        # what remains is the H3 cell, which the published figures already map.
        redacted = occ.drop(columns=[c for c in ("Latitude", "Longitude",
                                                 "Location")
                                     if c in occ.columns])
        redacted.to_csv(OUT / "reeps_h3.csv", index=False)
        for col in ("Latitude", "Longitude", "Location"):
            assert col not in redacted.columns, f"{col} leaked into reeps_h3.csv"
        assert len(redacted) == len(occ)
        print(f"  wrote reeps_h3.csv ({len(redacted)} records, "
              f"point localities removed, {len(redacted.columns)} fields)")

        cells = sorted(occ["h3_index"].dropna().unique())
        cen = pd.DataFrame(
            [{"h3_index": c,
              "lat": round(h3.cell_to_latlng(c)[0], 5),
              "lon": round(h3.cell_to_latlng(c)[1], 5)} for c in cells])
        cen.to_csv(OUT / "h3_cell_centroids.csv", index=False)
        print(f"  wrote h3_cell_centroids.csv ({len(cen)} cells, "
              f"centroid only, cell edge ~0.46 km)")

        for col in ("Latitude", "Longitude"):
            assert col not in agg.columns, f"{col} leaked into the aggregate"
        print("  checked: no point coordinates in the aggregated table")

    n = 0
    for f in DERIVED:
        p = BASE / f
        if p.exists():
            shutil.copy2(p, OUT / f)
            n += 1
    print(f"\n  copied {n} cell-level derived tables")

    total = sum(f.stat().st_size for f in OUT.glob("*"))
    print(f"\nrelease/data: {len(list(OUT.glob('*')))} files, "
          f"{total/1e6:.1f} MB")
    if not full:
        print("point localities withheld — rerun with --full to include them")


if __name__ == "__main__":
    main()
