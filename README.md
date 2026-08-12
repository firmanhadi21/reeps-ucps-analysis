# REEPS UCPS analysis — code

Analysis code for *Spatiotemporal distribution and conservation prioritisation of
REEPS fauna in the Upper Cisokan Pumped Storage area* (Frontiers in Conservation
Science).

**This repository contains code only. The occurrence data are available from the
authors on reasonable request** — see *Data availability* below.

## Pipeline

`python run_pipeline.py` executes four stages in dependency order. Each stage owns
a distinct set of outputs, so the pipeline is idempotent.

| stage | script | writes |
|---|---|---|
| 1 | `generate_figure_csvs.py` | `h3_analysis`, `reeps_h3`, `h3_diversity`, `survey_coverage`, `cooccurrence_species`, `presence_absence`, `turnover_cell_period` |
| 2 | `compute_cooccurrence_significance.py` | `cooccurrence_pairs` (φ, Fisher *p*, FDR *p*), `cooccurrence_significance_per_cell` |
| 3 | `compute_cpi_pca6.py` | `priority_index`, `cpi_correlation_matrix`, `pca_cpi_weights` |
| 4 | `compute_connectivity_corridors.py` | `h3_connectivity_full`, `gap_analysis`, `stepping_stones` |

Order matters: stage 3 needs the φ coefficients from stage 2, and stage 4 needs
the priority tiers from stage 3.

Supporting scripts: `compute_h3_resolution_comparison.py` (evidence for the
resolution choice), `prepare_data_release.py` (builds an aggregated data package),
and `script/build_*.py` (figures).

**All four stages require the occurrence database**, which is not distributed
here. Running the pipeline therefore requires obtaining the data first.

## Verifying a run

The pipeline is self-checking. Stage 1 reports record and cell counts; stage 4
runs validation gates. A correct run against the analysed dataset reports:

```
493 records | 11 species | 38 occupied cells | 149 grid cells | 111 unsurveyed
occupied patches: 1        keystone cells: 3
ALL GATES PASS
```

Any deviation means the inputs are not the analysed dataset, and the run fails
visibly rather than silently.

## Environment

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

PROJ and GDAL data come from the installed `rasterio` wheel. If your shell exports
`PROJ_LIB` pointing at another GIS installation, override it — a mismatched PROJ
database raises `CRSError: The EPSG code is unknown`.

## Analytical decisions recorded in the code

- **Species set.** `MANUSCRIPT_SPECIES` in `generate_figure_csvs.py` declares the
  11 species of Table 1 in one place. The `Status` field in the source database
  does not isolate this set, so filtering must be by explicit species list.
- **Data correction.** Record 366 (*Nycticebus javanicus*, 2014) carried a
  longitude roughly six degrees west of the study area — a digit transposition
  that placed it some 667 km away. The same named locality appears in the 2017
  survey at the correct longitude. The corrected value is rounded here, as this is
  a Critically Endangered species record.
- **Grid extent.** One H3 cell, drawn in by that erroneous coordinate, is excluded
  via `AOI_EXCLUDE_CELLS`, giving the 149-cell grid that tiles the AOI.
- **CPI components.** Co-occurrence is the mean φ coefficient across species pairs
  present in a cell, and the threat component counts Critically Endangered species
  only. Earlier formulations (pair count, and all CR+EN+VU taxa) were near-collinear
  with richness by construction.
- **Legacy layers.** `REEPS_GridAnalyses.gpkg` is retained with the source data for
  provenance but is **not authoritative**: it was built from a superseded
  565-record extract, and its diversity and priority layers do not correspond to
  the analysed set.

## Data availability

The occurrence dataset is **available from the authors on reasonable request**.

It is not published openly because 214 of the 493 analysed records (43.4%) are of
three Critically Endangered species subject to illegal collection and trade —
Javan Leopard (*Panthera pardus melas*), Javan Slow Loris (*Nycticebus javanicus*)
and Sunda Pangolin (*Manis javanica*). Because every analysis is conducted at the
level of H3 resolution-8 cells (~0.74 km², ~740 m across), even cell-aggregated
occurrence tables would localise these species finely enough to be of use for
targeting. Withholding them follows standard practice for threatened-species
records.

Requests should state the intended use and will be considered together with the
data owners (PLN and Perhutani). `prepare_data_release.py` builds the aggregated
package supplied to approved requesters.

## Citation

To be completed on publication.
