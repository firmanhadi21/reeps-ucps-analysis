"""
Generate intermediate CSV files needed by the static figure scripts.
Reads from rebuilt GeoPackages and produces the CSVs.
"""

# ── Portable project root (auto-inserted; replaces a hardcoded absolute path) ──
import os as _os
from pathlib import Path as _Path


def _reeps_base() -> _Path:
    """Locate the project root: $REEPS_BASE, else walk up to the marker file."""
    env = _os.environ.get("REEPS_BASE")
    if env:
        return _Path(env).expanduser().resolve()
    here = _Path(__file__).resolve()
    for cand in (here.parent, *here.parents):
        if (cand / "REEPS_Master_Database.gpkg").exists():
            return cand
    return here.parent


REEPS_BASE = _reeps_base()
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import geopandas as gpd
from collections import Counter
from math import log
import h3
import os

BASE = str(REEPS_BASE)
GRID_GPKG = os.path.join(BASE, 'REEPS_GridAnalyses.gpkg')
MASTER_GPKG = os.path.join(BASE, 'REEPS_Master_Database.gpkg')

# The 11 species reported in Table 1 of the manuscript. This is the analysed
# set: filtering the master occurrences to it yields exactly 493 records across
# 39 occupied H3 cells, matching every per-species value in that table.
# ── Data corrections ──────────────────────────────────────────────────────────
# Record No. 366 was
# transcribed with a longitude ~6 degrees west of the AOI — a 7->1 typo in the
# third digit that placed it 667 km west of the AOI, in the sea off Sumatra.
# The same survey locality appears in a later survey year at the corrected
# longitude, with latitude agreeing to three decimals. The value here is rounded
# for public release because the record is of a Critically Endangered taxon; the
# unrounded correction is held with the source database.
# Left uncorrected, this single record adds a spurious 39th "occupied" cell and
# breaks the single-contiguous-patch result reported in the manuscript.
COORDINATE_CORRECTIONS = {
    366: {"Longitude": 107.221},   # rounded to ~110 m for public release
}

# The same typo also pulled its H3 cell into the AOI grid. That hexagon lies far
# outside the ~110 km2 area of interest and is excluded so the grid is the 149
# cells that actually tile the AOI.
AOI_EXCLUDE_CELLS = {"888ce55b25fffff"}

MANUSCRIPT_SPECIES = {
    'Panthera pardus melas', 'Nycticebus javanicus', 'Manis javanica',
    'Hylobates moloch', 'Presbytis comata', 'Trachypithecus auratus',
    'Nisaetus bartelsi', 'Aonyx cinereus', 'Tragulus kanchil',
    'Prionailurus bengalensis', 'Paradoxurus hermaphroditus',
}

COMMON_NAMES = {
    'Panthera pardus melas': 'Javan Leopard',
    'Nycticebus javanicus': 'Javan Slow Loris',
    'Manis javanica': 'Sunda Pangolin',
    'Hylobates moloch': 'Javan Gibbon',
    'Presbytis comata': 'Grizzled Langur',
    'Trachypithecus auratus': 'Javan Langur',
    'Nisaetus bartelsi': "Bartels's Hawk-eagle",
    'Aonyx cinereus': 'Asian Small-clawed Otter',
    'Tragulus kanchil': 'Lesser Mouse-deer',
    'Prionailurus bengalensis': 'Leopard Cat',
    'Paradoxurus hermaphroditus': 'Common Palm Civet',
    'Hystrix javanica': 'Javan Porcupine',
    'Pteropus vampyrus': 'Large Flying Fox',
    'Arctictis binturong': 'Binturong',
    'Herpestes javanicus': 'Javan Mongoose',
}


def main():
    print("Generating intermediate CSVs for figure scripts...")

    # Load data
    h3_all = gpd.read_file(GRID_GPKG, layer='h3_all_cells')
    h3_rich = gpd.read_file(GRID_GPKG, layer='h3_richness_summary')
    h3_div = gpd.read_file(GRID_GPKG, layer='h3_diversity')
    h3_pri = gpd.read_file(GRID_GPKG, layer='h3_priority')
    h3_temp = gpd.read_file(GRID_GPKG, layer='h3_temporal_traj')
    h3_tmat = gpd.read_file(GRID_GPKG, layer='h3_temporal_matrix')
    h3_chao = gpd.read_file(GRID_GPKG, layer='h3_chao1')
    h3_corr = gpd.read_file(GRID_GPKG, layer='h3_corridor_gaps')
    h3_iso = gpd.read_file(GRID_GPKG, layer='h3_isolation_risk')
    h3_sgap = gpd.read_file(GRID_GPKG, layer='h3_survey_gaps')
    gdf_occ = gpd.read_file(MASTER_GPKG, layer='reeps_occurrences')

    # apply documented coordinate corrections, then re-derive the H3 cell
    for rec_no, fixes in COORDINATE_CORRECTIONS.items():
        mask = gdf_occ['No'] == rec_no
        if not mask.any():
            print(f"  WARNING: correction target record No={rec_no} not found")
            continue
        for col, val in fixes.items():
            old = gdf_occ.loc[mask, col].iloc[0]
            gdf_occ.loc[mask, col] = val
            print(f"  corrected record No={rec_no}: {col} {old} -> {val}")
        gdf_occ.loc[mask, 'h3_index'] = [
            h3.latlng_to_cell(r.Latitude, r.Longitude, 8)
            for r in gdf_occ.loc[mask].itertuples()
        ]

    # drop grid cells that lie outside the AOI
    for name, frame in (('h3_all', h3_all), ('h3_rich', h3_rich)):
        n = int(frame['h3_index'].isin(AOI_EXCLUDE_CELLS).sum())
        if n:
            print(f"  excluded {n} out-of-AOI cell(s) from {name}")
    h3_all = h3_all[~h3_all['h3_index'].isin(AOI_EXCLUDE_CELLS)]
    h3_rich = h3_rich[~h3_rich['h3_index'].isin(AOI_EXCLUDE_CELLS)]
    # Status == 'REEPS' tags 524 of the 596 master rows, so it does NOT isolate
    # the analysed set: it lets through non-target taxa and common-name entries
    # ('peusing', 'Biawak', 'Elang Hitam', 'Callosciurus sp.'). Filtering on the
    # explicit Table 1 species list is what reproduces the reported
    # 493 records / 39 occupied cells.
    reeps_occ = gdf_occ[
        (gdf_occ['Status'] == 'REEPS')
        & gdf_occ['Species'].isin(MANUSCRIPT_SPECIES)
    ].copy()
    n_cells = reeps_occ['h3_index'].nunique()
    print(f"  filtered to {len(MANUSCRIPT_SPECIES)} manuscript species: "
          f"{len(reeps_occ)} records, {n_cells} occupied cells "
          f"(expected 493 / 38 after the No=366 coordinate correction)")

    all_years = sorted(reeps_occ['Year'].dropna().unique())

    # ── 1. h3_analysis.csv — master grid summary (all cells) ──
    # Scripts expect snake_case: total_records, species_richness, trend_direction, etc.
    df = h3_rich.drop(columns=['geometry']).copy()
    df = df.rename(columns={
        'Total_Records': 'total_records', 'Species_Richness': 'species_richness',
        'Species_List': 'species_list', 'Trend_Direction': 'trend_direction',
        'Trend_Slope': 'trend_slope', 'First_Year': 'first_year',
        'Last_Year': 'last_year', 'Years_w__Data': 'years_w_data',
    })
    # Rename Records_YYYY columns
    for c in list(df.columns):
        if c.startswith('Records_'):
            df = df.rename(columns={c: c.lower()})
    df = df.merge(h3_all.drop(columns=['geometry'])[['h3_index', 'Occupied', 'Patch_ID',
                   'Occ__Nbrs', 'K2_Occ__Nbrs', 'Sp__Reachable_K2', 'Eco__Score']],
                  on='h3_index', how='left')
    df = df.rename(columns={
        'Occupied': 'occupied', 'Patch_ID': 'patch_id',
        'Occ__Nbrs': 'occ_nbrs', 'K2_Occ__Nbrs': 'k2_occ_nbrs',
        'Sp__Reachable_K2': 'sp_reachable_k2', 'Eco__Score': 'eco_score',
    })
    # Overwrite the per-cell occupancy columns with values recomputed from the
    # filtered occurrences. As shipped these came from the grid GeoPackage,
    # which counts non-target taxa and so marks 41 cells occupied instead of 39.
    occ_n = reeps_occ.groupby('h3_index').size()
    occ_s = reeps_occ.groupby('h3_index')['Species'].nunique()
    occ_l = reeps_occ.groupby('h3_index')['Species'].apply(
        lambda x: ', '.join(sorted(set(x))))
    df['total_records'] = df['h3_index'].map(occ_n).fillna(0).astype(int)
    df['species_richness'] = df['h3_index'].map(occ_s).fillna(0).astype(int)
    if 'species_list' in df.columns:
        df['species_list'] = df['h3_index'].map(occ_l).fillna('')
    df['occupied'] = (df['total_records'] > 0).astype(int)
    df.to_csv(os.path.join(BASE, 'h3_analysis.csv'), index=False)
    print(f"  h3_analysis.csv: {len(df)} rows "
          f"({int(df['occupied'].sum())} occupied, "
          f"{int(df['total_records'].sum())} records)")

    # ── 2. reeps_h3.csv — REEPS occurrences with h3 ──
    reeps_out = reeps_occ.drop(columns=['geometry']).copy()
    reeps_out.to_csv(os.path.join(BASE, 'reeps_h3.csv'), index=False)
    print(f"  reeps_h3.csv: {len(reeps_out)} rows")

    # ── 3. h3_diversity.csv ──
    # Recomputed from the filtered occurrences rather than copied out of
    # REEPS_GridAnalyses.gpkg: that layer was built from a 565-row occurrence
    # vintage and reproduces only 15 of its 41 rows, so its indices do not
    # correspond to the analysed 493-record set.
    stale_div = h3_div.drop(columns=['geometry']).rename(columns={
        'Diversity_Trend': 'diversity_trend', 'Trend_Slope': 'diversity_trend_slope',
    })
    trend_cols = [c for c in ('h3_index', 'diversity_trend', 'diversity_trend_slope')
                  if c in stale_div.columns]

    div_rows = []
    for cell, cell_df in reeps_occ.groupby('h3_index'):
        counts = Counter(cell_df['Species'])
        n = int(sum(counts.values()))
        s = len(counts)
        p = [c / n for c in counts.values()]
        shannon = -sum(x * log(x) for x in p if x > 0)
        simpson_D = sum(x * x for x in p)          # dominance
        dom_sp, dom_n = counts.most_common(1)[0]
        lat, lon = h3.cell_to_latlng(cell)
        div_rows.append({
            'h3_index': cell, 'lat': lat, 'lon': lon,
            'richness_S': s, 'records_N': n,
            'shannon_H': shannon,
            'simpson_D': 1.0 - simpson_D,          # reported as Simpson's 1-D
            'pielou_J': shannon / log(s) if s > 1 else 0.0,
            'berger_parker_BP': dom_n / n,
            'dominant_species': dom_sp,
            'dominant_common': COMMON_NAMES.get(dom_sp, dom_sp),
        })
    div_out = pd.DataFrame(div_rows).sort_values('h3_index').reset_index(drop=True)
    if len(trend_cols) > 1:
        div_out = div_out.merge(stale_div[trend_cols], on='h3_index', how='left')
    div_out.to_csv(os.path.join(BASE, 'h3_diversity.csv'), index=False)
    print(f"  h3_diversity.csv: {len(div_out)} rows "
          f"(mean H'={div_out['shannon_H'].mean():.3f}, "
          f"mean 1-D={div_out['simpson_D'].mean():.3f})")

    # ── 4. priority_index.csv ──
    # Scripts expect both PascalCase and snake_case columns
    pri_out = h3_pri.drop(columns=['geometry']).copy()
    # Normalize scores to 0-1 for score_ columns
    for src, dst in [('Richness', 'score_richness'), ('Diversity', 'score_diversity'),
                     ('Connectivity', 'score_connectivity'),
                     ('Co_occurrence', 'score_cooccurrence'),
                     ('Threatened', 'score_threatened')]:
        vmin, vmax = pri_out[src].min(), pri_out[src].max()
        pri_out[dst] = (pri_out[src] - vmin) / (vmax - vmin) if vmax > vmin else 0
    # Keep Priority_Index and Priority_Tier as PascalCase (scripts expect this)
    pri_out = pri_out.rename(columns={'Tier': 'Priority_Tier'})
    # priority_index.csv is owned by compute_cpi_pca6.py, which builds the
    # six-component PCA-weighted index described in the Methods. The copy that
    # used to be written here came from the grid GeoPackage and was the
    # five-component expert-weighted index, so writing it would overwrite the
    # real one whenever this script ran afterwards.
    print("  priority_index.csv: skipped (owned by compute_cpi_pca6.py)")

    # ── 5. h3_connectivity_full.csv ──
    conn = h3_all.drop(columns=['geometry']).copy()
    conn = conn.rename(columns={
        'Cell_Type': 'cell_type', 'Occupied': 'occupied', 'Patch_ID': 'patch_id',
        'Occ__Nbrs': 'occ_nbrs', 'K2_Occ__Nbrs': 'k2_occ_nbrs',
        'Sp__Reachable_K2': 'sp_reachable_k2', 'Eco__Score': 'eco_score',
    })
    # owned by compute_connectivity_corridors.py
    print("  h3_connectivity_full.csv: skipped "
          "(owned by compute_connectivity_corridors.py)")

    # ── 6. survey_coverage.csv ──
    # Per-cell survey coverage
    rows = []
    for cell in h3_rich['h3_index']:
        cell_df = reeps_occ[reeps_occ['h3_index'] == cell]
        n_methods = cell_df['Survey_Method'].nunique() if len(cell_df) > 0 else 0
        methods = ', '.join(sorted(cell_df['Survey_Method'].dropna().unique())) if len(cell_df) > 0 else ''
        n_years = cell_df['Year'].nunique() if len(cell_df) > 0 else 0
        n_rec = len(cell_df)
        # Effort tier based on records
        if n_rec >= 10:
            tier = 'High'
        elif n_rec >= 3:
            tier = 'Medium'
        elif n_rec >= 1:
            tier = 'Low'
        else:
            tier = 'None'
        rows.append({
            'h3_index': cell,
            'H3_Index': cell,
            'total_records': n_rec,
            'n_methods': n_methods,
            'methods': methods,
            'n_years': n_years,
            'species_richness': cell_df['Species'].nunique() if n_rec > 0 else 0,
            'Effort_Tier': tier,
        })
    pd.DataFrame(rows).to_csv(os.path.join(BASE, 'survey_coverage.csv'), index=False)
    print(f"  survey_coverage.csv: {len(rows)} rows")

    # ── 7. gap_analysis.csv ──
    gap = h3_sgap.drop(columns=['geometry']).copy()
    gap = gap.rename(columns={
        'K1_Occ_Neighbors': 'k1_occ_neighbors', 'K2_Occ_Neighbors': 'k2_occ_neighbors',
        'Dist_to_Nearest_Occ_km': 'dist_to_nearest_occ_km',
        'K1_Nbr_Richness': 'k1_nbr_richness', 'K2_Nbr_Richness': 'k2_nbr_richness',
        'Corridor_Candidate': 'corridor_candidate',
        'Gap_Priority_Score': 'gap_priority_score', 'Survey_Priority': 'survey_priority',
    })
    # owned by compute_connectivity_corridors.py
    print("  gap_analysis.csv: skipped "
          "(owned by compute_connectivity_corridors.py)")

    # ── 8. stepping_stones.csv ──
    ss = h3_corr.drop(columns=['geometry']).copy()
    ss = ss.rename(columns={
        'Betweenness': 'betweenness', 'K1_Occ_Neighbors': 'k1_occ_neighbors',
        'K2_Richness_Reachable': 'k2_richness_reachable',
        'Dist_to_Nearest_Occ': 'dist_to_nearest_occ',
        'Is_Corridor_Candidate': 'is_corridor_candidate',
        'Connects_Priority_Cells': 'connects_priority_cells',
        'Stepping_Stone_Score': 'stepping_stone_score', 'Corridor_Tier': 'corridor_tier',
    })
    # owned by compute_connectivity_corridors.py
    print("  stepping_stones.csv: skipped "
          "(owned by compute_connectivity_corridors.py)")

    # ── 9. cooccurrence_pairs.csv ──
    species = sorted(reeps_occ['Species'].unique())
    presence = {}
    for sp in species:
        presence[sp] = set(reeps_occ[reeps_occ['Species'] == sp]['h3_index'])
    pairs = []
    for i in range(len(species)):
        for j in range(i+1, len(species)):
            sp1, sp2 = species[i], species[j]
            shared = len(presence[sp1] & presence[sp2])
            union = len(presence[sp1] | presence[sp2])
            jaccard = shared / union if union > 0 else 0
            pairs.append({
                'species_1': sp1, 'species_2': sp2,
                'common_1': COMMON_NAMES.get(sp1, ''),
                'common_2': COMMON_NAMES.get(sp2, ''),
                'shared_cells': shared,
                'union_cells': union,
                'jaccard': round(jaccard, 4),
                'cells_1': len(presence[sp1]),
                'cells_2': len(presence[sp2]),
            })
    # owned by compute_cooccurrence_significance.py, which adds the phi
    # coefficient, Fisher p-values and FDR-adjusted p-values
    print("  cooccurrence_pairs.csv: skipped "
          "(owned by compute_cooccurrence_significance.py)")

    # ── 10. cooccurrence_species.csv (per-species summary) ──
    sp_summary = []
    for sp in species:
        n_cooc = sum(1 for p in pairs
                     if (p['species_1'] == sp or p['species_2'] == sp) and p['shared_cells'] > 0)
        sp_summary.append({
            'species': sp, 'common_name': COMMON_NAMES.get(sp, ''),
            'n_cells': len(presence[sp]),
            'n_cooccurring_species': n_cooc,
            'total_records': len(reeps_occ[reeps_occ['Species'] == sp]),
        })
    pd.DataFrame(sp_summary).to_csv(os.path.join(BASE, 'cooccurrence_species.csv'), index=False)
    print(f"  cooccurrence_species.csv: {len(sp_summary)} rows")

    # ── 11. presence_absence.csv (species x cell matrix) ──
    cells = sorted(reeps_occ['h3_index'].unique())
    pa_data = {'species': species}
    for cell in cells:
        cell_sp = set(reeps_occ[reeps_occ['h3_index'] == cell]['Species'])
        pa_data[cell] = [1 if sp in cell_sp else 0 for sp in species]
    pd.DataFrame(pa_data).to_csv(os.path.join(BASE, 'presence_absence.csv'), index=False)
    print(f"  presence_absence.csv: {len(species)} species x {len(cells)} cells")

    # ── 12. turnover_cell_period.csv ──
    rows = []
    occupied_cells = reeps_occ['h3_index'].unique()
    for cell in occupied_cells:
        cell_df = reeps_occ[reeps_occ['h3_index'] == cell]
        present_years = sorted(cell_df['Year'].dropna().unique())
        if len(present_years) < 2:
            continue
        for i in range(1, len(present_years)):
            y1, y2 = present_years[i-1], present_years[i]
            sp1 = set(cell_df[cell_df['Year'] == y1]['Species'])
            sp2 = set(cell_df[cell_df['Year'] == y2]['Species'])
            gained = len(sp2 - sp1)
            lost = len(sp1 - sp2)
            shared = len(sp1 & sp2)
            union = len(sp1 | sp2)
            beta = (gained + lost) / union if union > 0 else 0
            rows.append({
                'h3_index': cell, 'period': f'{int(y1)}-{int(y2)}',
                'year_from': int(y1), 'year_to': int(y2),
                'sp_from': len(sp1), 'sp_to': len(sp2),
                'shared': shared, 'gained': gained, 'lost': lost,
                'beta_whittaker': round(beta, 4),
            })
    pd.DataFrame(rows).to_csv(os.path.join(BASE, 'turnover_cell_period.csv'), index=False)
    print(f"  turnover_cell_period.csv: {len(rows)} rows")

    print("\nDone! All intermediate CSVs generated.")


if __name__ == '__main__':
    main()
