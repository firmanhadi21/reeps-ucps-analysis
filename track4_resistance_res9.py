#!/usr/bin/env python3
"""
Dual-resolution resistance connectivity:
- Res 9 (~0.11 km²) for resistance surface and corridor modelling
- Aggregate results back to res 8 for reporting alongside biodiversity metrics
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


import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import Polygon, mapping
from shapely.ops import transform
import pyproj
import networkx as nx
import h3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir(str(REEPS_BASE))

# ============================================================
# 1. GENERATE RES-9 CELLS FOR THE AOI
# ============================================================
print("="*60)
print("1. GENERATING RES-9 H3 CELLS")
print("="*60)

# The res-8 grid itself also comes from h3_analysis.csv. The GeoPackage layer holds
# 150 cells, one of which is the cell drawn in by the mistyped longitude; using it
# generated 150 x 7 = 1,050 res-9 children for an AOI that contains 149 cells.
# Reads the refreshed GeoPackages built by build_geopackages.py: the originals
# hold the unfiltered 596-record database on a 150-cell grid, not the analysed
# 493 records across 11 species on 149 cells.
gdf_r8 = gpd.read_file("REEPS_GridAnalyses_v2.gpkg", layer="h3_all_cells")

# Occupancy and richness come from h3_analysis.csv, not from the GeoPackage.
#
# The GeoPackage layers hold the unfiltered database — 596 records across 26 taxa in
# 41 cells — rather than the analysed set of 493 records across the 11 REEPS species
# in 38 cells. Taking the occupied set from there marked three extra res-8 parents as
# occupied, one of them the spurious cell produced by the mistyped longitude, and so
# seeded the corridor analysis from 21 res-9 cells that hold no REEPS record.
_h3a = pd.read_csv("h3_analysis.csv")
occupied_r8 = set(_h3a.loc[_h3a["total_records"] > 0, "h3_index"])
richness_r8 = dict(zip(_h3a["h3_index"], _h3a["species_richness"]))

# Restrict the grid to the 149 AOI cells the analysis uses.
_before = len(gdf_r8)
gdf_r8 = gdf_r8[gdf_r8["h3_index"].isin(set(_h3a["h3_index"]))].copy()
print(f"Res-8 grid: {_before} cells in the GeoPackage -> {len(gdf_r8)} in the AOI")
print(f"Occupied res-8 cells (from h3_analysis.csv): {len(occupied_r8)}")

# Generate res-9 children for every res-8 cell
r9_cells = []
r9_to_r8 = {}
for h8 in gdf_r8['h3_index']:
    children = h3.cell_to_children(h8, 9)
    for c in children:
        r9_cells.append(c)
        r9_to_r8[c] = h8

print(f"Res-8 cells: {len(gdf_r8)}")
print(f"Res-9 cells: {len(r9_cells)}")
print(f"Children per parent: {len(r9_cells) / len(gdf_r8):.0f}")

# Build res-9 geodataframe
project_to_utm = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32748', always_xy=True).transform

r9_geoms = []
for c in r9_cells:
    boundary = h3.cell_to_boundary(c)
    coords = [(lng, lat) for lat, lng in boundary]
    coords.append(coords[0])
    poly = Polygon(coords)
    poly_utm = transform(project_to_utm, poly)
    r9_geoms.append(poly_utm)

gdf_r9 = gpd.GeoDataFrame({
    'h3_r9': r9_cells,
    'h3_r8_parent': [r9_to_r8[c] for c in r9_cells],
    'is_occupied_r8': [r9_to_r8[c] in occupied_r8 for c in r9_cells],
    'richness_r8': [richness_r8.get(r9_to_r8[c], 0) for c in r9_cells]
}, geometry=r9_geoms, crs='EPSG:32748')

print(f"Res-9 cells in occupied res-8 parents: {gdf_r9['is_occupied_r8'].sum()}")

# ============================================================
# 2. ASSIGN RESISTANCE PER RES-9 CELL
# ============================================================
print("\n" + "="*60)
print("2. RESISTANCE PER RES-9 CELL")
print("="*60)

class_names = {
    1: 'Waterbody', 2: 'Paddy', 3: 'Built-up', 4: 'Natural Forest',
    5: 'Production Forest', 6: 'Agroforest', 7: 'Young Regeneration',
    8: 'Crop Cultivation', 9: 'Sparse Vegetation', 10: 'Bareland', 11: 'Clouds'
}

class_resistance = {
    0: 50, 1: 90, 2: 50, 3: 100, 4: 1, 5: 5, 6: 5,
    7: 10, 8: 40, 9: 30, 10: 80, 11: 50
}

lc_raster = 'cisokan-spatial/PS_Final_11class_Hierarchical.tif'

with rasterio.open(lc_raster) as src:
    print(f"Land cover raster: {src.width}x{src.height}, {src.res[0]}m")

    resistances = []
    dominant_classes = []
    pct_forests = []

    for idx, row in gdf_r9.iterrows():
        try:
            geom = [mapping(row.geometry)]
            out, _ = rio_mask(src, geom, crop=True, filled=True, nodata=0)
            data = out[0]
            valid = data[(data >= 1) & (data <= 10)]

            if len(valid) > 0:
                vals, counts = np.unique(valid, return_counts=True)
                total_px = counts.sum()
                mean_r = sum(class_resistance.get(int(v), 50) * c for v, c in zip(vals, counts)) / total_px
                dominant = int(vals[np.argmax(counts)])
                forest_px = sum(c for v, c in zip(vals, counts) if v in [4, 5, 6, 7])
                pct_forest = forest_px / total_px * 100
            else:
                mean_r = 50
                dominant = 0
                pct_forest = 0
        except:
            mean_r = 50
            dominant = 0
            pct_forest = 0

        resistances.append(mean_r)
        dominant_classes.append(dominant)
        pct_forests.append(pct_forest)

    gdf_r9['resistance'] = resistances
    gdf_r9['dominant_class'] = dominant_classes
    gdf_r9['dominant_name'] = [class_names.get(d, 'NoData') for d in dominant_classes]
    gdf_r9['pct_forest'] = pct_forests

print(f"\nRes-9 resistance stats:")
print(f"  Range: {gdf_r9['resistance'].min():.1f} - {gdf_r9['resistance'].max():.1f}")
print(f"  Mean: {gdf_r9['resistance'].mean():.1f}")
print(f"  SD: {gdf_r9['resistance'].std():.1f}")
print(f"\nRes-9 dominant land cover:")
print(gdf_r9['dominant_name'].value_counts().head(10).to_string())

# Compare with res-8 heterogeneity
r9_by_parent = gdf_r9.groupby('h3_r8_parent')['resistance'].agg(['mean', 'std', 'min', 'max'])
print(f"\nWithin-res-8 resistance heterogeneity (across res-9 children):")
print(f"  Mean within-parent SD: {r9_by_parent['std'].mean():.1f}")
print(f"  Max within-parent range: {(r9_by_parent['max'] - r9_by_parent['min']).max():.1f}")

# ============================================================
# 3. BUILD RES-9 RESISTANCE GRAPH
# ============================================================
print("\n" + "="*60)
print("3. BUILDING RES-9 RESISTANCE GRAPH")
print("="*60)

resist_lookup = dict(zip(gdf_r9['h3_r9'], gdf_r9['resistance']))
r9_set = set(r9_cells)

G = nx.Graph()
for c in r9_cells:
    G.add_node(c)

edge_count = 0
for c in r9_cells:
    neighbors = set(h3.grid_disk(c, 1))
    neighbors.discard(c)
    for nb in neighbors:
        if nb in r9_set and not G.has_edge(c, nb):
            r1 = resist_lookup.get(c, 50)
            r2 = resist_lookup.get(nb, 50)
            G.add_edge(c, nb, weight=(r1 + r2) / 2)
            edge_count += 1

print(f"Res-9 graph: {G.number_of_nodes()} nodes, {edge_count} edges")

# ============================================================
# 4. CORRIDOR BETWEENNESS AT RES-9
# ============================================================
print("\n" + "="*60)
print("4. CORRIDOR BETWEENNESS AT RES-9")
print("="*60)

# Source nodes: all res-9 cells within occupied res-8 parents
sources = [c for c in r9_cells if r9_to_r8[c] in occupied_r8]
gap_cells = [c for c in r9_cells if r9_to_r8[c] not in occupied_r8]

print(f"Source cells (occupied): {len(sources)}")
print(f"Gap cells (unoccupied): {len(gap_cells)}")

# Use NetworkX betweenness centrality subset for efficiency
# Compute betweenness only for paths between source nodes
print("Computing betweenness centrality (this may take a moment)...")
bc = nx.betweenness_centrality_subset(G, sources=sources, targets=sources, weight='weight')

gdf_r9['bc_resist'] = gdf_r9['h3_r9'].map(bc)

# Also compute unit-weight betweenness for comparison
G_unit = G.copy()
for u, v in G_unit.edges():
    G_unit[u][v]['weight'] = 1
bc_unit = nx.betweenness_centrality_subset(G_unit, sources=sources, targets=sources, weight='weight')
gdf_r9['bc_unit'] = gdf_r9['h3_r9'].map(bc_unit)

# Top gap cells
gap_df = gdf_r9[~gdf_r9['is_occupied_r8']].copy()
top_resist = gap_df.nlargest(10, 'bc_resist')[['h3_r9', 'h3_r8_parent', 'resistance', 'dominant_name', 'bc_resist']]
top_unit = gap_df.nlargest(10, 'bc_unit')[['h3_r9', 'h3_r8_parent', 'resistance', 'dominant_name', 'bc_unit']]

print("\nTop 10 corridor cells (resistance-weighted):")
for _, r in top_resist.iterrows():
    print(f"  {r['h3_r9']} (parent {r['h3_r8_parent']}): BC={r['bc_resist']:.4f}, R={r['resistance']:.1f}, {r['dominant_name']}")

print("\nTop 10 corridor cells (unit-weight):")
for _, r in top_unit.iterrows():
    print(f"  {r['h3_r9']} (parent {r['h3_r8_parent']}): BC={r['bc_unit']:.4f}, R={r['resistance']:.1f}, {r['dominant_name']}")

# ============================================================
# 5. AGGREGATE TO RES-8
# ============================================================
print("\n" + "="*60)
print("5. AGGREGATING TO RES-8")
print("="*60)

r8_agg = gdf_r9.groupby('h3_r8_parent').agg(
    mean_resistance=('resistance', 'mean'),
    sd_resistance=('resistance', 'std'),
    min_resistance=('resistance', 'min'),
    max_resistance=('resistance', 'max'),
    pct_forest=('pct_forest', 'mean'),
    max_bc_resist=('bc_resist', 'max'),
    mean_bc_resist=('bc_resist', 'mean'),
    max_bc_unit=('bc_unit', 'max'),
    mean_bc_unit=('bc_unit', 'mean'),
).reset_index()

# Spearman correlation between unit and resistance betweenness at res-8 level
from scipy.stats import spearmanr
gap_r8 = r8_agg[~r8_agg['h3_r8_parent'].isin(occupied_r8)]
rho, p = spearmanr(gap_r8['max_bc_unit'], gap_r8['max_bc_resist'])
print(f"Res-8 aggregated corridor betweenness:")
print(f"  Spearman rho (unit vs resist): {rho:.3f} (p = {p:.4f})")

# Top-15 overlap
top15_unit = set(gap_r8.nlargest(15, 'max_bc_unit')['h3_r8_parent'])
top15_resist = set(gap_r8.nlargest(15, 'max_bc_resist')['h3_r8_parent'])
overlap = len(top15_unit & top15_resist)
print(f"  Top-15 overlap: {overlap}/15")

r8_agg.to_csv('resistance_res9_aggregated_to_res8.csv', index=False)

# ============================================================
# 6. FIGURES
# ============================================================
print("\n" + "="*60)
print("6. GENERATING FIGURES")
print("="*60)

gdf_r9_wgs = gdf_r9.to_crs(epsg=4326)
gdf_r8_wgs = gdf_r8.to_crs(epsg=4326)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Res-9 resistance surface
ax = axes[0, 0]
gdf_r9_wgs.plot(column='resistance', cmap='RdYlGn_r', edgecolor='none',
                ax=ax, legend=True, vmin=1, vmax=100,
                legend_kwds={'label': 'Resistance (1-100)', 'shrink': 0.6})
gdf_r8_wgs[gdf_r8_wgs['h3_index'].isin(occupied_r8)].plot(
    edgecolor='blue', linewidth=1.5, facecolor='none', ax=ax)
ax.set_title('(A) Resistance Surface (Res-9, ~0.11 km²)\nPlanetScope 11-class + Withaningsih et al. 2022',
             fontsize=10, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=8)
ax.set_ylabel('Latitude', fontsize=8)
ax.tick_params(labelsize=7)
# Panel A legend
handles_a = [
    mpatches.Patch(facecolor='#006400', edgecolor='black', label='Low resistance (forest)'),
    mpatches.Patch(facecolor='#FFFF00', edgecolor='black', label='Medium resistance'),
    mpatches.Patch(facecolor='#FF0000', edgecolor='black', label='High resistance (built-up)'),
    mpatches.Patch(facecolor='none', edgecolor='blue', linewidth=1.5, label='Occupied res-8 cell'),
]
ax.legend(handles=handles_a, loc='lower left', fontsize=6.5, framealpha=0.90,
          edgecolor='#CCCCCC', handlelength=1.2, labelspacing=0.25)

# Panel B: Res-9 corridor betweenness (resistance-weighted)
ax = axes[0, 1]
gap_wgs = gdf_r9_wgs[~gdf_r9_wgs['is_occupied_r8']].copy()
gap_wgs.plot(column='bc_resist', cmap='YlOrRd', edgecolor='none',
             ax=ax, legend=True,
             legend_kwds={'label': 'Betweenness', 'shrink': 0.6})
gdf_r9_wgs[gdf_r9_wgs['is_occupied_r8']].plot(
    facecolor='lightblue', edgecolor='none', alpha=0.5, ax=ax)
ax.set_title('(B) Corridor Betweenness (Res-9)\nResistance-weighted',
             fontsize=10, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=8)
ax.tick_params(labelsize=7)
# Panel B legend
handles_b = [
    mpatches.Patch(facecolor='#FEB24C', edgecolor='black', label='High betweenness (YlOrRd)'),
    mpatches.Patch(facecolor='lightblue', edgecolor='none', alpha=0.5, label='Occupied res-8 parent'),
]
ax.legend(handles=handles_b, loc='lower left', fontsize=6.5, framealpha=0.90,
          edgecolor='#CCCCCC', handlelength=1.2, labelspacing=0.25)

# Panel C: Res-9 dominant land cover
ax = axes[1, 0]
lc_colors = {
    'Natural Forest': '#064a00', 'Production Forest': '#228b22',
    'Agroforest': '#8b4513', 'Young Regeneration': '#7cfc00',
    'Crop Cultivation': '#e35c22', 'Paddy': '#12faff',
    'Sparse Vegetation': '#90EE90', 'Built-up': '#ffff00',
    'Bareland': '#e90f50', 'Waterbody': '#0e59d6', 'NoData': '#cccccc'
}
gdf_r9_wgs['lc_color'] = gdf_r9_wgs['dominant_name'].map(
    lambda x: lc_colors.get(x, '#cccccc'))
gdf_r9_wgs.plot(color=gdf_r9_wgs['lc_color'], edgecolor='none', ax=ax)
# Legend for top classes
top_classes = gdf_r9['dominant_name'].value_counts().head(6).index
patches = [mpatches.Patch(color=lc_colors.get(c, '#ccc'), label=c) for c in top_classes]
ax.legend(handles=patches, loc='lower left', fontsize=7, title='Land Cover')
ax.set_title('(C) Dominant Land Cover per Res-9 Cell\nPlanetScope Hierarchical Classification',
             fontsize=10, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=8)
ax.set_ylabel('Latitude', fontsize=8)
ax.tick_params(labelsize=7)

# Panel D: Res-8 aggregated — resistance SD (heterogeneity)
ax = axes[1, 1]
gdf_r8_plot = gdf_r8_wgs.merge(r8_agg, left_on='h3_index', right_on='h3_r8_parent', how='left')
gdf_r8_plot.plot(column='sd_resistance', cmap='Purples', edgecolor='black',
                 linewidth=0.3, ax=ax, legend=True,
                 legend_kwds={'label': 'Resistance SD', 'shrink': 0.6})
gdf_r8_plot[gdf_r8_plot['h3_index'].isin(occupied_r8)].plot(
    edgecolor='blue', linewidth=1.5, facecolor='none', ax=ax)
ax.set_title('(D) Within-Cell Resistance Heterogeneity\n(SD across Res-9 children per Res-8 cell)',
             fontsize=10, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=8)
ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('figures/resistance_res9_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/resistance_res9_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure saved to figures/resistance_res9_analysis.pdf")

# Save res-9 results
gdf_r9[['h3_r9', 'h3_r8_parent', 'resistance', 'dominant_name', 'pct_forest',
         'bc_resist', 'bc_unit', 'is_occupied_r8']].to_csv(
    'resistance_res9_results.csv', index=False)
print("Results saved to: resistance_res9_results.csv")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("MANUSCRIPT SUMMARY")
print("="*60)
print(f"""
Dual-resolution resistance analysis:
  Res-9 cells: {len(r9_cells)} (~0.11 km² each)
  Land cover: {gdf_r9['dominant_name'].value_counts().head(4).to_dict()}
  Mean resistance: {gdf_r9['resistance'].mean():.1f} (SD {gdf_r9['resistance'].std():.1f})
  Within-res-8 resistance SD: {r9_by_parent['std'].mean():.1f} (justifies res-9 for connectivity)

Corridor betweenness (res-8 aggregated):
  Spearman rho (unit vs resist): {rho:.3f}
  Top-15 corridor overlap: {overlap}/15
""")
