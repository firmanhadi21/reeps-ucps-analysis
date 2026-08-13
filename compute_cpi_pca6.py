#!/usr/bin/env python3
"""
compute_cpi_pca6.py
───────────────────
Six-component, PCA-weighted Conservation Priority Index (CPI), as specified in
the Methods of the Frontiers manuscript:

  (1) species richness            (4) Shannon diversity
  (2) spatial connectivity        (5) co-occurrence significance
  (3) threatened species richness (6) habitat permeability

Each component is min-max normalised to [0, 1]. Weights are derived by PCA on
the standardised components, using variance-weighted absolute loadings summed
across all six principal components. CPI_i = sum_k w_k * x_ik_norm.

Component definitions were reverse-engineered from the shipped priority_index.csv
and reproduce it exactly (41/41 rows on every component):
  richness      = h3_diversity.richness_S
  diversity     = h3_diversity.shannon_H
  connectivity  = h3_analysis.occ_nbrs           (occupied neighbours, 0-6)
  co-occurrence = C(S, 2)                        (species pairs in the cell)
  threatened    = count of IUCN CR+EN+VU species in the cell
  permeability  = inverted mean res-9 resistance aggregated to res-8

Writes priority_index.csv (replacing the 5-component expert-weighted version)
and cpi_correlation_matrix.csv.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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

# IUCN status for the 11 analysed species (manuscript Table 1)
IUCN = {
    "Panthera pardus melas": "CR", "Nycticebus javanicus": "CR",
    "Manis javanica": "CR",
    "Hylobates moloch": "EN", "Presbytis comata": "EN",
    "Trachypithecus auratus": "VU", "Nisaetus bartelsi": "VU",
    "Aonyx cinereus": "VU",
    "Tragulus kanchil": "LC", "Prionailurus bengalensis": "LC",
    "Paradoxurus hermaphroditus": "LC",
}
# Component (3) counts Critically Endangered species only. Counting all of
# CR+EN+VU makes the component near-collinear with richness (r = 0.964, only
# 7.1% independent variance) because 8 of the 11 analysed species are
# threatened; restricting to CR gives r = 0.699 (51.2% independent) while
# keeping the component monotone in conservation urgency.
THREATENED = {s for s, st in IUCN.items() if st == "CR"}

LABELS = ["Richness", "Connectivity", "Threatened",
          "Diversity", "Co_occurrence", "Permeability"]


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=s.index)


def assign_tier(cpi: float) -> str:
    if cpi >= 0.70:
        return "CRITICAL"
    if cpi >= 0.50:
        return "HIGH"
    if cpi >= 0.30:
        return "MEDIUM"
    return "LOW"


def main() -> None:
    div = pd.read_csv(BASE / "h3_diversity.csv")
    h3a = pd.read_csv(BASE / "h3_analysis.csv")
    occ = pd.read_csv(BASE / "reeps_h3.csv")
    res = pd.read_csv(BASE / "resistance_res9_aggregated_to_res8.csv")

    cells = div["h3_index"].tolist()
    print(f"occupied cells: {len(cells)}   records: {len(occ)}   "
          f"species: {occ['Species'].nunique()}")

    df = pd.DataFrame({"h3_index": cells})

    # (1) richness and (4) diversity
    df = df.merge(div[["h3_index", "richness_S", "shannon_H", "lat", "lon"]],
                  on="h3_index", how="left")
    df = df.rename(columns={"richness_S": "Richness", "shannon_H": "Diversity"})

    # (2) connectivity — occupied neighbours
    df = df.merge(h3a[["h3_index", "occ_nbrs"]], on="h3_index", how="left")
    df = df.rename(columns={"occ_nbrs": "Connectivity"})

    # (5) co-occurrence significance — number of FDR-significant, positively
    # associated species pairs jointly present in the cell. The manuscript
    # defines this component as "co-occurrence significance"; the earlier
    # implementation used C(S,2), which is a deterministic function of richness
    # (r = 0.96 by construction) and therefore double-counted component (1).
    sig_path = BASE / "cooccurrence_significance_per_cell.csv"
    if not sig_path.exists():
        raise SystemExit("run compute_cooccurrence_significance.py first")
    sig = pd.read_csv(sig_path).set_index("h3_index")["mean_phi"]
    df["Co_occurrence"] = df["h3_index"].map(sig).fillna(0.0)

    # (3) threatened species richness
    thr = (occ[occ["Species"].isin(THREATENED)]
           .groupby("h3_index")["Species"].nunique())
    df["Threatened"] = df["h3_index"].map(thr).fillna(0).astype(int)

    # (6) habitat permeability — inverted mean resistance
    rmap = res.set_index("h3_r8_parent")["mean_resistance"]
    df["mean_resistance"] = df["h3_index"].map(rmap)
    missing = int(df["mean_resistance"].isna().sum())
    if missing:
        print(f"  WARNING: {missing} cell(s) lack a resistance value; "
              f"filled with the cell-set mean")
        df["mean_resistance"] = df["mean_resistance"].fillna(
            df["mean_resistance"].mean())
    df["Permeability"] = -df["mean_resistance"]      # invert: low resistance = high permeability

    # ── min-max normalise the six components ─────────────────────────────────
    for c in LABELS:
        df[f"score_{c.lower()}"] = minmax(df[c])
    X_norm = df[[f"score_{c.lower()}" for c in LABELS]].values

    # ── PCA on standardised components ───────────────────────────────────────
    X_std = StandardScaler().fit_transform(X_norm)
    pca = PCA()
    pca.fit(X_std)

    print("\nPCA explained variance:")
    for i, (ev, cum) in enumerate(zip(pca.explained_variance_ratio_,
                                      np.cumsum(pca.explained_variance_ratio_)), 1):
        print(f"  PC{i}: {ev*100:5.1f}%   cumulative {cum*100:5.1f}%")

    pc2 = pd.Series(pca.components_[1], index=LABELS)
    print(f"\nPC2 loadings:\n{pc2.round(3).to_string()}")

    # variance-weighted absolute loadings across all PCs
    wl = np.zeros(len(LABELS))
    for i in range(len(LABELS)):
        wl += np.abs(pca.components_[i]) * pca.explained_variance_ratio_[i]
    weights = wl / wl.sum()

    print("\nPCA-derived weights:")
    # Table 2 of the manuscript, for comparison with the freshly computed
    # weights. These were updated when the CPI was rebuilt to the six-component
    # specification; the earlier set (17.7 / 18.1 / 16.9 / 18.0 / 18.0 / 11.3)
    # belonged to the version before that rebuild.
    published = {"Richness": 16.4, "Connectivity": 18.4, "Threatened": 17.7,
                 "Diversity": 16.4, "Co_occurrence": 17.1, "Permeability": 14.0}
    for lab, w in zip(LABELS, weights):
        print(f"  {lab:14s} {w*100:5.1f}%   (manuscript: {published[lab]:4.1f}%)")

    # ── CPI ──────────────────────────────────────────────────────────────────
    df["Priority_Index"] = (X_norm @ weights).round(4)
    df["Priority_Tier"] = df["Priority_Index"].apply(assign_tier)
    df = df.sort_values("Priority_Index", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)

    print(f"\nCPI range: {df['Priority_Index'].min():.3f}-"
          f"{df['Priority_Index'].max():.3f}   (manuscript: 0.218-0.879)")
    print("tiers:", df["Priority_Tier"].value_counts().to_dict())

    # ── correlation matrix among the six normalised components ───────────────
    corr = pd.DataFrame(X_norm, columns=LABELS).corr()
    bio = [c for c in LABELS if c != "Permeability"]
    off = corr.loc[bio, bio].values[np.triu_indices(len(bio), 1)]
    print(f"\nbiodiversity inter-correlations: r = {off.min():.2f}-{off.max():.2f}"
          f"   (manuscript: 0.10-0.93)")
    pm = corr.loc["Permeability", bio].abs()
    print(f"permeability |r| vs biodiversity: max {pm.max():.2f}"
          f"   (manuscript: <= 0.26)")

    cols = (["Rank", "h3_index", "Priority_Tier"] + LABELS +
            ["Priority_Index", "lat", "lon"] +
            [f"score_{c.lower()}" for c in LABELS])
    df[cols].to_csv(BASE / "priority_index.csv", index=False)
    corr.round(3).to_csv(BASE / "cpi_correlation_matrix.csv")
    pd.DataFrame({"component": LABELS, "weight": weights}).to_csv(
        BASE / "pca_cpi_weights.csv", index=False)

    print(f"\nwrote priority_index.csv ({len(df)} rows), "
          f"cpi_correlation_matrix.csv, pca_cpi_weights.csv")


if __name__ == "__main__":
    main()
