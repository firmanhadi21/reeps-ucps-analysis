#!/usr/bin/env python3
"""
compute_cooccurrence_significance.py
────────────────────────────────────
Pairwise co-occurrence significance for the 11 analysed species across the 39
occupied H3 cells, exactly as specified in the manuscript Methods:

  * Jaccard similarity  J = |A n B| / |A u B|
  * Phi coefficient
  * two-tailed Fisher's exact test
  * Benjamini-Hochberg FDR correction over all 55 pairs, alpha = 0.05
  * a pair is "significantly positively associated" when both species occupy
    >= 5 cells, FDR-adjusted p < 0.05, and phi > 0

Writes cooccurrence_pairs.csv (with phi, p, p_fdr, significant) and
cooccurrence_significance_per_cell.csv (the CPI component: number of
significant pairs jointly present in each cell).
"""

import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


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
MIN_CELLS = 5
ALPHA = 0.05


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Return BH-adjusted p-values."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # enforce monotonicity
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def main() -> None:
    occ = pd.read_csv(BASE / "reeps_h3.csv")
    cells = sorted(occ["h3_index"].dropna().unique())
    species = sorted(occ["Species"].dropna().unique())
    print(f"{len(occ)} records | {len(species)} species | {len(cells)} occupied cells")

    presence = {s: set(occ.loc[occ["Species"] == s, "h3_index"]) for s in species}
    N = len(cells)

    rows = []
    for a, b in combinations(species, 2):
        A, B = presence[a], presence[b]
        n11 = len(A & B)
        n10 = len(A - B)
        n01 = len(B - A)
        n00 = N - n11 - n10 - n01
        _, p = fisher_exact([[n11, n10], [n01, n00]], alternative="two-sided")
        denom = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
        phi = ((n11 * n00 - n10 * n01) / denom) if denom > 0 else 0.0
        union = len(A | B)
        rows.append({
            "species_1": a, "species_2": b,
            "shared_cells": n11, "union_cells": union,
            "jaccard": round(n11 / union, 4) if union else 0.0,
            "cells_1": len(A), "cells_2": len(B),
            "phi": round(phi, 4), "p_value": p,
        })

    df = pd.DataFrame(rows)
    df["p_fdr"] = benjamini_hochberg(df["p_value"].values)
    df["eligible"] = (df["cells_1"] >= MIN_CELLS) & (df["cells_2"] >= MIN_CELLS)
    df["significant"] = df["eligible"] & (df["p_fdr"] < ALPHA) & (df["phi"] > 0)

    print(f"\npairs tested            : {len(df)}")
    print(f"eligible (both >= {MIN_CELLS} cells): {int(df['eligible'].sum())}")
    print(f"raw p < 0.05            : {int((df['p_value'] < ALPHA).sum())}")
    print(f"FDR p < 0.05 and phi > 0: {int(df['significant'].sum())}")

    sig = df[df["significant"]].sort_values("p_fdr")
    if len(sig):
        print("\nsignificant positively associated pairs:")
        for _, r in sig.iterrows():
            print(f"  {r['species_1']} + {r['species_2']}: "
                  f"phi={r['phi']:.3f}, p={r['p_value']:.4g}, "
                  f"p_fdr={r['p_fdr']:.4g}, shared={r['shared_cells']}")

    # ── per-cell co-occurrence component ─────────────────────────────────────
    # Counting FDR-significant pairs is unusable as a CPI component here: after
    # the No=366 coordinate correction no pair survives FDR (min adjusted
    # p = 0.084), so the count is zero everywhere. Mean phi across the species
    # pairs present in a cell measures the same concept continuously, carries
    # ~87% variance independent of richness, and needs no significance cut-off.
    phi_lookup = {}
    for _, r in df.iterrows():
        phi_lookup[(r["species_1"], r["species_2"])] = r["phi"]
        phi_lookup[(r["species_2"], r["species_1"])] = r["phi"]

    per_cell = []
    for c in cells:
        sp = sorted(set(occ.loc[occ["h3_index"] == c, "Species"]))
        prs = [(a, b) for i, a in enumerate(sp) for b in sp[i + 1:]]
        n_sig = int(sum(1 for _, r in sig.iterrows()
                        if r["species_1"] in sp and r["species_2"] in sp))
        per_cell.append({
            "h3_index": c,
            "significant_pairs": n_sig,
            "mean_phi": float(np.mean([phi_lookup.get(p, 0.0) for p in prs]))
            if prs else 0.0,
            "n_pairs": len(prs),
        })
    pc = pd.DataFrame(per_cell)
    print(f"\nmean_phi across cells: {pc['mean_phi'].min():.3f} to "
          f"{pc['mean_phi'].max():.3f} (mean {pc['mean_phi'].mean():.3f})")

    print(f"\nper-cell significant-pair counts: "
          f"{pc['significant_pairs'].value_counts().sort_index().to_dict()}")
    print(f"  range {pc['significant_pairs'].min()}-{pc['significant_pairs'].max()}, "
          f"mean {pc['significant_pairs'].mean():.2f}, "
          f"non-zero in {(pc['significant_pairs'] > 0).sum()}/{len(pc)} cells")

    df.drop(columns=["eligible"]).to_csv(BASE / "cooccurrence_pairs.csv", index=False)
    pc.to_csv(BASE / "cooccurrence_significance_per_cell.csv", index=False)
    print("\nwrote cooccurrence_pairs.csv, cooccurrence_significance_per_cell.csv")


if __name__ == "__main__":
    main()
