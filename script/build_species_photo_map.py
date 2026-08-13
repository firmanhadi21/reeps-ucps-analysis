#!/usr/bin/env python3
"""
build_species_photo_map.py
──────────────────────────
Reviewer 4 suggested that species photographs would sit well within the maps.
This builds that figure: the distribution of each photographed species across the
occupied H3 cells, with the survey photograph of that species inset beside it.

Only species for which a usable photograph exists in the 2011/2012 survey reports
are shown — five of the eleven analysed species. No photograph of the three
Critically Endangered species is available at usable resolution: the reports
record them by tracks, scat and burrows, and the one camera-trap frame captioned
Panthera pardus melas shows no discernible animal.

Writes figures/fig_species_photos.{pdf,png}
"""

import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPoly
from PIL import Image
from shapely.geometry import Polygon

import h3


def _reeps_base() -> Path:
    here = Path(__file__).resolve()
    for cand in (here.parent, *here.parents):
        if (cand / "REEPS_Master_Database.gpkg").exists():
            return cand
    return here.parent


BASE = _reeps_base()
sys.path.insert(0, str(BASE / "script"))

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.linewidth": 0.5,
    "figure.dpi": 300, "savefig.dpi": 300,
})

# species, photo file, IUCN, whether the photo is a night camera-trap frame
SPECIES = [
    ("Hylobates moloch", "Javan Gibbon", "EN",
     "hylobates_moloch_gibbon.png", False),
    ("Presbytis comata", "Javan Surili", "EN",
     "presbytis_comata_surili.png", False),
    ("Trachypithecus auratus", "Javan Lutung", "VU",
     "trachypithecus_auratus_langur_1.png", False),
    ("Prionailurus bengalensis", "Leopard Cat", "LC",
     "prionailurus_bengalensis_leopardcat.png", True),
    ("Paradoxurus hermaphroditus", "Common Palm Civet", "LC",
     "paradoxurus_hermaphroditus_civet.png", True),
]
IUCN_C = {"EN": "#E65100", "VU": "#F9A825", "LC": "#558B2F"}


def hex_poly(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def main() -> None:
    occ = pd.read_csv(BASE / "reeps_h3.csv")
    h3a = pd.read_csv(BASE / "h3_analysis.csv")
    photos = BASE / "species_photos"

    all_cells = h3a["h3_index"].tolist()
    occupied = set(h3a.loc[h3a["species_richness"] > 0, "h3_index"])
    geom_all = {c: hex_poly(c) for c in all_cells}

    try:
        from basemap_utils import S2_RGBA, S2_EXTENT, add_graticule
        have_bm = True
    except Exception as e:
        print(f"  basemap unavailable ({e})")
        have_bm = False

    FIG_W, FIG_H = 14.0, 5.4
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    xs = [g.exterior.xy[0] for g in geom_all.values()]
    ys = [g.exterior.xy[1] for g in geom_all.values()]
    x0, x1 = min(min(x) for x in xs), max(max(x) for x in xs)
    y0, y1 = min(min(y) for y in ys), max(max(y) for y in ys)
    padx, pady = (x1 - x0) * 0.04, (y1 - y0) * 0.08

    # one column per species: distribution map above, photograph below
    N = len(SPECIES)
    LEFT, RIGHT, COL_GAP = 0.020, 0.012, 0.012
    COL_W = (1.0 - LEFT - RIGHT - (N - 1) * COL_GAP) / N
    MAP_Y, MAP_H = 0.560, 0.365
    PHOTO_TOP, PHOTO_H_MAX = 0.505, 0.315

    for i, (sci, common, iucn, fn, is_trap) in enumerate(SPECIES):
        bx = LEFT + i * (COL_W + COL_GAP)

        axm = fig.add_axes([bx, MAP_Y, COL_W, MAP_H])
        sp_cells = set(occ.loc[occ["Species"] == sci, "h3_index"])
        if have_bm:
            axm.imshow(S2_RGBA, extent=S2_EXTENT, origin="upper", zorder=0)
        for c, g in geom_all.items():
            if c in sp_cells:
                fc, a, lw = IUCN_C[iucn], 0.82, 0.35
            elif c in occupied:
                fc, a, lw = "#FFFFFF", 0.16, 0.25
            else:
                continue
            xy = np.column_stack(g.exterior.xy)
            axm.add_patch(MplPoly(xy, closed=True, facecolor=fc, alpha=a,
                                  edgecolor="#37474F", linewidth=lw, zorder=3))
        axm.set_xlim(x0 - padx, x1 + padx)
        axm.set_ylim(y0 - pady, y1 + pady)
        # The five panels are narrow, so only the leftmost is labelled; the rest
        # carry the graticule lines alone to keep the strip readable.
        if have_bm:
            add_graticule(axm, nx=2, ny=2, xlabels=(i == 0), ylabels=(i == 0),
                          fontsize=5.0)
        else:
            axm.set_xticks([]); axm.set_yticks([])
        for sp_ in axm.spines.values():
            sp_.set_edgecolor("#9E9E9E")
        axm.set_title(f"$\\it{{{sci.replace(' ', chr(92) + ' ')}}}$",
                      loc="left", fontsize=8.2, pad=3)
        axm.text(0.03, 0.04, f"{len(sp_cells)} of {len(occupied)} cells",
                 transform=axm.transAxes, fontsize=6.8,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                           alpha=0.85))

        pth = photos / fn
        if not pth.exists():
            continue
        img = Image.open(pth)
        iw, ih = img.size
        # fit inside the column, preserving aspect
        pw = COL_W
        ph = pw * (ih / iw) * (FIG_W / FIG_H)
        if ph > PHOTO_H_MAX:
            pw *= PHOTO_H_MAX / ph
            ph = PHOTO_H_MAX
        axp = fig.add_axes([bx + (COL_W - pw) / 2, PHOTO_TOP - ph, pw, ph])
        axp.imshow(img)
        axp.set_xticks([]); axp.set_yticks([])
        for sp_ in axp.spines.values():
            sp_.set_edgecolor("#616161")
        cap = f"{common} ({iucn})" + ("  · night camera trap" if is_trap else "")
        axp.set_title(cap, loc="left", fontsize=7.2, color=IUCN_C[iucn], pad=2.5)

    fig.text(0.020, 0.105,
             "Distribution of the five analysed species for which a survey "
             "photograph is available, each shown against the 38 occupied H3 "
             "cells (pale outlines); cells where the species was detected are "
             "filled in the colour of its\nIUCN category. Photographs are from "
             "the thematic surveys of the UCPS project area (LIPI 2012; Survei "
             "II 2012). No photograph of the three Critically Endangered species "
             "was available at usable\nresolution; those taxa are recorded in the "
             "survey reports by tracks, scat and burrows.",
             fontsize=7.2, color="#424242", va="top")

    out = BASE / "figures"
    for ext in ("pdf", "png"):
        f = out / f"fig_species_photos.{ext}"
        fig.savefig(f, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  wrote {f.name} ({f.stat().st_size/1e6:.1f} MB)")
    plt.close(fig)


if __name__ == "__main__":
    main()
