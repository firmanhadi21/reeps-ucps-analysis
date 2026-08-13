#!/usr/bin/env python3
"""Rebuild every figure the manuscript includes, in dependency order.

Scope: this regenerates the GeoPackages and all figures from the CSV analysis
outputs. It does NOT re-run the upstream pipeline that produces those CSVs
(generate_figure_csvs.py and the compute_* analysis scripts), because the values
reported in the manuscript were verified against the current CSVs; regenerating
them is a separate, deliberate step.

Some generators write straight into figures/, others write into a static_figures/
subtree under their own names and are copied to the name the manuscript includes.
That mapping is the point of this script: a figure regenerated but not copied is
exactly how figures/fig3_diversity fell a step behind its source.
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

import os
import runpy
import shutil
import sys
import traceback

# The shell profile points PROJ_LIB at an OTB install whose proj.db is too old for
# rasterio; clearing it lets pyproj fall back to its own bundled database.
for _v in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_v, None)

B = str(REEPS_BASE)
S = os.path.join(B, "script")
os.chdir(B)
if S not in sys.path:
    sys.path.insert(0, S)

# (script, [(produced_stem, manuscript_stem), ...])  empty list = writes in place
JOBS = [
    ("build_geopackages.py", []),

    ("script/build_fig1_study_area.py", []),
    ("script/build_h3_static_figures.py",
     [("static_figures/h3_map/fig_H3_E_composite", "fig2_h3")]),
    ("script/build_diversity_figures.py",
     [("diversity_figures/fig_E_diversity_composite", "fig3_diversity")]),
    ("script/build_conn_static_figures.py",
     [("static_figures/connectivity/fig_Conn_E_composite", "fig4_connectivity")]),
    ("script/build_cooc_static_figures.py",
     [("static_figures/cooccurrence/fig_CoOc_E_composite", "fig5_cooccurrence")]),
    ("script/build_priority_static_figures.py",
     [("static_figures/priority/fig_Prior_E_composite", "fig6_priority")]),
    ("session4_species_maps.py", []),
    ("script/build_species_photo_map.py", []),
    ("script/build_corridor_static_figures.py",
     [("static_figures/corridor/fig_Corr_E_composite", "fig8_corridor")]),
    ("track4_resistance_res9.py", []),
    ("script/build_reserves_corridor_figure.py", []),

    ("compute_morans_i.py", []),
    ("compute_lisa.py", []),
    ("compute_accumulation_rarefaction.py", []),
    ("script/build_cpi_permeability_figure.py", []),
]


def run(script: str) -> bool:
    print(f"\n{'=' * 72}\n{script}\n{'=' * 72}", flush=True)
    try:
        sys.argv = [os.path.basename(script)]
        runpy.run_path(os.path.join(B, script), run_name="__main__")
        return True
    except Exception:
        traceback.print_exc()
        print(f"  !! {script} FAILED", flush=True)
        return False


def main() -> None:
    failed, copied = [], 0
    for script, mapping in JOBS:
        if not run(script):
            failed.append(script)
            continue
        for produced, target in mapping:
            for ext in (".pdf", ".png"):
                src = os.path.join(B, produced + ext)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(B, "figures", target + ext))
                    copied += 1
                    print(f"  -> figures/{target}{ext}", flush=True)
                else:
                    print(f"  !! missing output: {produced}{ext}", flush=True)
                    failed.append(f"{script} ({produced}{ext})")

    print(f"\n{'=' * 72}")
    print(f"{len(JOBS) - len(failed)}/{len(JOBS)} scripts ok, {copied} files copied")
    if failed:
        print("FAILURES:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    print("ALL FIGURES REBUILT")


if __name__ == "__main__":
    main()
