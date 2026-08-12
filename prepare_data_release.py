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
        agg = (occ.groupby(["h3_index", "Species", "Year"])
               .size().reset_index(name="records"))
        agg.to_csv(OUT / "reeps_h3_aggregated.csv", index=False)
        print(f"\n  wrote reeps_h3_aggregated.csv "
              f"({len(agg)} species-cell-period rows, no coordinates)")

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
