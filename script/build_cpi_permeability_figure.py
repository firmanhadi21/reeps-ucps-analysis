#!/usr/bin/env python3
"""
build_cpi_permeability_figure.py
────────────────────────────────
Regenerates figures/cpi_permeability_comparison.{pdf,png} — the PCA-based CPI
component analysis (Figure S5).

The shipped version of this figure had no generator anywhere in the codebase, so
its panel values could not be reproduced or updated. This script rebuilds all
three panels from the current six-component index:

  (A) CPI computed with and without the permeability component
  (B) PCA-derived variance-weighted loadings for the six components
  (C) species richness against habitat permeability
"""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
LAB = ["Richness", "Connectivity", "Threatened", "Diversity",
       "Co_occurrence", "Permeability"]
PRETTY = {"Co_occurrence": "Co-occurrence", "Diversity": "Shannon"}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 0.6,
})

TIER_C = {"CRITICAL": "#B71C1C", "HIGH": "#EF6C00",
          "MEDIUM": "#FBC02D", "LOW": "#7CB342"}


def tiers(c):
    return np.select([c >= 0.70, c >= 0.50, c >= 0.30],
                     ["CRITICAL", "HIGH", "MEDIUM"], "LOW")


def main() -> None:
    pri = pd.read_csv(BASE / "priority_index.csv")
    w = pd.read_csv(BASE / "pca_cpi_weights.csv").set_index("component")["weight"]
    corr = pd.read_csv(BASE / "cpi_correlation_matrix.csv", index_col=0)

    X = pri[[f"score_{c.lower()}" for c in LAB]].values
    cpi = X @ w[LAB].values
    t_full = tiers(cpi)

    keep = [i for i, c in enumerate(LAB) if c != "Permeability"]
    w5 = w[[LAB[i] for i in keep]].values
    w5 = w5 / w5.sum()
    cpi_np = X[:, keep] @ w5

    r = np.corrcoef(cpi, cpi_np)[0, 1]
    agree = (tiers(cpi_np) == t_full).mean()
    r_rp = corr.loc["Richness", "Permeability"]
    print(f"(A) r = {r:.3f}, tier agreement {agree:.1%}")
    print(f"(B) weights {', '.join(f'{c} {w[c]*100:.1f}%' for c in LAB)}")
    print(f"(C) richness vs permeability r = {r_rp:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

    # ── (A) ─────────────────────────────────────────────────────────────────
    ax = axes[0]
    for t, col in TIER_C.items():
        m = t_full == t
        if m.any():
            ax.scatter(cpi_np[m], cpi[m], s=34, c=col, edgecolor="#37474F",
                       linewidth=0.4, label=t, zorder=3)
    lim = [min(cpi.min(), cpi_np.min()) - 0.05, max(cpi.max(), cpi_np.max()) + 0.05]
    ax.plot(lim, lim, ls="--", c="#9E9E9E", lw=0.9, zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("CPI without permeability")
    ax.set_ylabel("CPI (six components)")
    ax.set_title("(a)  Sensitivity to the permeability component", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.text(0.04, 0.95, f"$r$ = {r:.3f}\n{agree:.1%} tier agreement",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDBDBD", lw=0.5))

    # ── (B) ─────────────────────────────────────────────────────────────────
    ax = axes[1]
    vals = [w[c] * 100 for c in LAB]
    cols = ["#455A64"] * 5 + ["#00838F"]
    bars = ax.bar([PRETTY.get(c, c) for c in LAB], vals, color=cols,
                  edgecolor="#263238", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%",
                ha="center", fontsize=8)
    ax.set_ylabel("PCA-derived weight (%)")
    ax.set_ylim(0, max(vals) + 3)
    ax.set_title("(b)  Variance-weighted PCA loadings", loc="left")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    # ── (C) ─────────────────────────────────────────────────────────────────
    ax = axes[2]
    ax.scatter(pri["Richness"], pri["Permeability"], s=34, c="#00838F",
               edgecolor="#37474F", linewidth=0.4, zorder=3)
    ax.set_xlabel("Species richness (S)")
    ax.set_ylabel("Habitat permeability (inverted resistance)")
    ax.set_title("(c)  Richness against permeability", loc="left")
    ax.text(0.04, 0.95, f"$r$ = {r_rp:.2f}", transform=ax.transAxes,
            va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDBDBD", lw=0.5))

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = BASE / "figures"
    for ext in ("pdf", "png"):
        p = out / f"cpi_permeability_comparison.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  wrote {p.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
