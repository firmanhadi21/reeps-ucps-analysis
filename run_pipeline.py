#!/usr/bin/env python3
"""
run_pipeline.py
───────────────
Runs the REEPS analysis pipeline in dependency order. Each stage owns a
distinct set of output files, so running this end to end is idempotent:

  1. generate_figure_csvs.py            h3_analysis, reeps_h3, h3_diversity,
                                        survey_coverage, cooccurrence_species,
                                        presence_absence, turnover_cell_period
  2. compute_cooccurrence_significance  cooccurrence_pairs (+ phi, p, FDR),
                                        cooccurrence_significance_per_cell
  3. compute_cpi_pca6.py                priority_index, cpi_correlation_matrix,
                                        pca_cpi_weights
  4. compute_connectivity_corridors.py  h3_connectivity_full, gap_analysis,
                                        stepping_stones

Order matters: stage 3 needs stage 2's phi values, and stage 4 needs stage 3's
priority tiers.
"""

import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY = BASE / ".venv/bin/python"
if not PY.exists():
    PY = Path(sys.executable)

STAGES = [
    "generate_figure_csvs.py",
    "compute_cooccurrence_significance.py",
    "compute_cpi_pca6.py",
    "compute_connectivity_corridors.py",
]

env = dict(os.environ)
proj = BASE / ".venv/lib/python3.13/site-packages/rasterio/proj_data"
gdal = BASE / ".venv/lib/python3.13/site-packages/rasterio/gdal_data"
if proj.exists():
    env.update(PROJ_LIB=str(proj), PROJ_DATA=str(proj), GDAL_DATA=str(gdal))

for i, stage in enumerate(STAGES, 1):
    print(f"\n{'='*70}\n[{i}/{len(STAGES)}] {stage}\n{'='*70}")
    r = subprocess.run([str(PY), str(BASE / stage)], cwd=BASE, env=env)
    if r.returncode != 0:
        print(f"\nFAILED at stage {i}: {stage} (exit {r.returncode})")
        sys.exit(r.returncode)

print(f"\n{'='*70}\npipeline complete\n{'='*70}")
