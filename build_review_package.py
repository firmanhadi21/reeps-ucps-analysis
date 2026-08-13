#!/usr/bin/env python3
"""Build a single self-contained ZIP for reviewers: data + scripts + README.

Everything sits flat at the package root. That is deliberate: each script locates
the project by walking up for a marker file, and when the marker is absent it falls
back to the script's own directory. Flat means that fallback resolves to the package
root, so a reviewer can unzip and run without setting any environment variable.

Figure-generation scripts are excluded. They need the 917 MB Sentinel-2 scene, which
is not distributed, so shipping them would only produce failures.
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE / "review_package"
ZIP = BASE / "REEPS_review_package.zip"
DATA = BASE / "release" / "data"

# Analysis scripts a reviewer can actually run, plus stage 1 for completeness.
SCRIPTS = [
    "generate_figure_csvs.py",              # stage 1 — needs point data, see README
    "compute_cooccurrence_significance.py",  # stage 2
    "compute_cpi_pca6.py",                   # stage 3
    "compute_connectivity_corridors.py",     # stage 4
    "compute_morans_i.py",
    "compute_lisa.py",
    "compute_accumulation_rarefaction.py",
    "compute_h3_resolution_comparison.py",
    "run_pipeline.py",
    "requirements.txt",
]

SENSITIVE_COLS = ("Latitude", "Longitude", "Location")


def build() -> None:
    if not DATA.exists():
        sys.exit("release/data missing — run prepare_data_release.py first")

    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir()

    n_data = 0
    for f in sorted(DATA.iterdir()):
        if f.name == "README_FOR_REVIEWERS.md":
            shutil.copy2(f, STAGE / "README.md")
            continue
        shutil.copy2(f, STAGE / f.name)
        n_data += 1

    n_code = 0
    for name in SCRIPTS:
        src = BASE / name
        if not src.exists():
            print(f"  !! missing: {name}")
            continue
        shutil.copy2(src, STAGE / name)
        n_code += 1

    print(f"staged {n_data} data files, {n_code} scripts")


def verify() -> None:
    """No coordinates, no locality names, nothing unexpected."""
    import pandas as pd

    bad = []
    for f in sorted(STAGE.glob("*.csv")):
        cols = pd.read_csv(f, nrows=0).columns
        leaked = [c for c in SENSITIVE_COLS if c in cols]
        if leaked:
            bad.append((f.name, leaked))
    print(f"  coordinate/locality columns: {bad if bad else 'none'}")
    assert not bad, "sensitive column in the review package"

    # Check the scripts against the actual locality names in the source database
    # rather than a hardcoded list. Hardcoding them would put the names into this
    # file, which is itself published to the public code repository.
    src = BASE / "reeps_h3.csv"
    names = set()
    if src.exists():
        col = pd.read_csv(src).get("Location")
        if col is not None:
            names = {str(v).strip() for v in col.dropna().unique()
                     if len(str(v).strip()) > 4}
    hits = []
    for f in STAGE.glob("*.py"):
        txt = f.read_text(encoding="utf8", errors="replace")
        hits += [(f.name, n) for n in names if n in txt]
    assert not hits, f"a script names a survey locality: {hits}"
    print(f"  locality names in scripts: none (checked {len(names)} names)")

    occ = pd.read_csv(STAGE / "reeps_h3.csv")
    assert len(occ) == 493, f"reeps_h3.csv has {len(occ)} records, expected 493"
    assert occ["Species"].nunique() == 11
    print(f"  reeps_h3.csv: {len(occ)} records, "
          f"{occ['Species'].nunique()} species, "
          f"{occ['h3_index'].nunique()} cells")


def smoke_test() -> None:
    """Run the pipeline in a throwaway copy, with no REEPS_BASE set."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        box = Path(tmp) / "unzipped"
        shutil.copytree(STAGE, box)

        env = dict(os.environ)
        for v in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA", "REEPS_BASE"):
            env.pop(v, None)

        runnable = ["compute_cooccurrence_significance.py", "compute_cpi_pca6.py",
                    "compute_connectivity_corridors.py", "compute_morans_i.py",
                    "compute_lisa.py", "compute_accumulation_rarefaction.py"]
        failed = []
        for s in runnable:
            r = subprocess.run([sys.executable, s], cwd=box, env=env,
                               capture_output=True, text=True, timeout=900)
            status = "ok" if r.returncode == 0 else "FAILED"
            if r.returncode:
                failed.append((s, r.stderr.strip().split("\n")[-1][:110]))
            print(f"    {s:42s} {status}")
        if failed:
            for s, e in failed:
                print(f"      {s}: {e}")
            sys.exit("smoke test failed — package is not self-contained")


def zip_it() -> None:
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, Path("REEPS_review_package") / f.relative_to(STAGE))
    n = len(zipfile.ZipFile(ZIP).namelist())
    print(f"\nwrote {ZIP.name}  ({ZIP.stat().st_size/1024:.0f} KB, {n} files)")


if __name__ == "__main__":
    build()
    print("\nverifying:")
    verify()
    print("\nsmoke test (no REEPS_BASE, fresh copy):")
    smoke_test()
    zip_it()
