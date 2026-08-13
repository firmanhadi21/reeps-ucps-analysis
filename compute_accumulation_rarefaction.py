#!/usr/bin/env python3
"""Supplementary Figures S3 and S4 — species accumulation and rarefaction.

Rewritten in revision. Both figures were previously built from the GeoPackage layers,
which hold the unfiltered database (596 records, 26 taxa) rather than the analysed set
(493 records, 11 REEPS species). After the analysis-set filter was applied upstream the
figures were never rebuilt, so they showed 403 records against the 493 their captions
claim, stopped at the 2024 survey, and carried a working note ("excl. P. linsang") in
the S3 title. Both now read reeps_h3.csv, the same source as every other figure.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir(str(REEPS_BASE))

os.makedirs("figures", exist_ok=True)   # a fresh checkout has no figures/
SEED = 42
BOOTSTRAP = 500


def chao1(counts):
    """Classic Chao1 with its 95% log-transformed confidence interval."""
    counts = np.asarray([c for c in counts if c > 0], dtype=float)
    s_obs = len(counts)
    f1 = int((counts == 1).sum())
    f2 = int((counts == 2).sum())
    if f2 > 0:
        s_est = s_obs + f1 ** 2 / (2 * f2)
    else:
        s_est = s_obs + f1 * (f1 - 1) / 2
    if s_est <= s_obs:
        return s_obs, s_obs, s_obs
    var = 0.0
    if f2 > 0:
        a = f1 / f2
        var = f2 * (a ** 4 / 4 + a ** 3 + a ** 2 / 2)
    if var <= 0:
        return s_est, s_obs, s_est
    d = s_est - s_obs
    k = np.exp(1.96 * np.sqrt(np.log(1 + var / d ** 2)))
    return s_est, s_obs + d / k, s_obs + d * k


def rarefy(species, n_max, seed=SEED, reps=BOOTSTRAP):
    """Mean and 95% interval of expected richness against records sampled."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(species)
    xs = np.arange(1, n_max + 1)
    curves = np.zeros((reps, len(xs)))
    for r in range(reps):
        perm = rng.permutation(arr)
        seen, out = set(), np.zeros(len(xs))
        for i, sp in enumerate(perm):
            seen.add(sp)
            out[i] = len(seen)
        curves[r] = out
    return xs, curves.mean(axis=0), np.percentile(curves, [2.5, 97.5], axis=0)


def main() -> None:
    occ = pd.read_csv("reeps_h3.csv")
    occ["Year"] = pd.to_numeric(occ["Year"], errors="coerce")
    species = occ["Species"].tolist()
    n = len(species)
    s_obs = occ["Species"].nunique()
    print(f"records: {n}, species: {s_obs}")

    counts = occ["Species"].value_counts().values
    est, lo, hi = chao1(counts)
    print(f"Chao1 = {est:.1f} (95% CI {lo:.1f}-{hi:.1f}), observed {s_obs}")

    # ── S3: accumulation in record order, with the Chao1 asymptote ────────────
    seen, cum = set(), []
    for sp in species:
        seen.add(sp)
        cum.append(len(seen))

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(range(1, n + 1), cum, color="#1f4ed8", lw=2.2,
            label="Observed accumulation")
    ax.axhline(est, color="#d7191c", ls="--", lw=2,
               label=f"Chao1 = {est:.1f}")
    ax.set_xlabel("Cumulative records", fontsize=11)
    ax.set_ylabel("Cumulative species", fontsize=11)
    ax.set_title(f"Species accumulation with Chao1 asymptote "
                 f"({n} records, {s_obs} species)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(est, s_obs) + 2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig("figures/species_accumulation.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("figures/species_accumulation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote figures/species_accumulation.pdf and .png")

    # ── S4: individual-based rarefaction, pooled and per period ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    xs, mean, ci = rarefy(species, n)
    ax = axes[0]
    ax.plot(xs, mean, color="#1f4ed8", lw=2.2, label="Rarefaction (95% CI)")
    ax.fill_between(xs, ci[0], ci[1], color="#1f4ed8", alpha=0.18)
    ax.axhline(s_obs, color="#d7191c", ls="--", lw=2,
               label=f"Observed S = {s_obs}")
    at50 = mean[min(49, len(mean) - 1)]
    print(f"  rarefied richness at 50 records: {at50:.1f}")
    ax.set_title(f"(A) Individual-based rarefaction\n"
                 f"(all years pooled, n = {n})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of records sampled", fontsize=10)
    ax.set_ylabel("Expected species richness", fontsize=10)
    ax.set_ylim(0, s_obs + 3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    years = sorted(occ["Year"].dropna().unique())
    cmap = plt.get_cmap("viridis")
    for i, yr in enumerate(years):
        sub = occ.loc[occ["Year"] == yr, "Species"].tolist()
        if len(sub) < 2:
            continue
        xs_y, mean_y, _ = rarefy(sub, len(sub), reps=200)
        ax.plot(xs_y, mean_y, color=cmap(i / max(1, len(years) - 1)), lw=1.9,
                label=f"{int(yr)} (n={len(sub)})")
    ax.set_title("(B) Per-period rarefaction\n(sampling adequacy by survey year)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of records sampled", fontsize=10)
    ax.set_ylabel("Expected species richness", fontsize=10)
    ax.set_ylim(0, s_obs + 3)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)

    fig.tight_layout()
    fig.savefig("figures/rarefaction_curves.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("figures/rarefaction_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote figures/rarefaction_curves.pdf and .png")


if __name__ == "__main__":
    main()
