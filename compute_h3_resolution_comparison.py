#!/usr/bin/env python3
"""
compute_h3_resolution_comparison.py
───────────────────────────────────
Regenerates h3_resolution_comparison.csv from the audited 493-record dataset.
This table is the empirical basis for choosing H3 resolution 8, so it must
reflect the analysed records rather than an earlier unfiltered extract.
"""

import os
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
CELL_AREA_KM2 = {7: 5.16, 8: 0.74, 9: 0.11}


def main() -> None:
    occ = pd.read_csv(BASE / "reeps_h3.csv")
    print(f"{len(occ)} records, {occ['Species'].nunique()} species")

    rows = []
    for res in (7, 8, 9):
        cells = occ.apply(
            lambda r: h3.latlng_to_cell(r["Latitude"], r["Longitude"], res),
            axis=1)
        g = occ.assign(cell=cells).groupby("cell")["Species"].nunique()
        single = int((g == 1).sum())
        rows.append({
            "Resolution": res,
            "Cell_area_km2": CELL_AREA_KM2[res],
            "Occupied_cells": int(g.size),
            "Mean_richness": round(float(g.mean()), 3),
            "Max_richness": int(g.max()),
            "Single_species_cells": single,
            "Pct_single": round(100 * single / g.size, 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv(BASE / "h3_resolution_comparison.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nwrote h3_resolution_comparison.csv")


if __name__ == "__main__":
    main()
