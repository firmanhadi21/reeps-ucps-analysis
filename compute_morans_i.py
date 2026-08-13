#!/usr/bin/env python3
"""
Compute Global Moran's I for spatial autocorrelation of REEPS species richness
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from libpysal.weights import Queen
from esda.moran import Moran
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("figures", exist_ok=True)   # a fresh checkout has no figures/

# Load H3 grid data.
#
# This reads h3_analysis.csv rather than the GeoPackage layer it used previously.
# The GeoPackage was not refreshed after the coordinate correction, so it still
# carried the spurious 39th cell 667 km west of the AOI. That cell had no
# neighbours, which stretched the panel-A map across six degrees of longitude and
# put a zero in the denominator of the spatial lag, printing "R2 = nan" on panel B.
print("Loading H3 grid data...")
import h3 as _h3
from shapely.geometry import Polygon as _Polygon

_df = pd.read_csv("h3_analysis.csv")
_df = _df[_df["species_richness"] > 0].copy()
gdf = gpd.GeoDataFrame(
    _df,
    geometry=[_Polygon([(lon, lat) for lat, lon in _h3.cell_to_boundary(c)])
              for c in _df["h3_index"]],
    crs="EPSG:4326",
)

print(f"Loaded {len(gdf)} occupied H3 cells")

richness_col = 'species_richness'

print(f"\nUsing richness column: {richness_col}")
print(f"Richness range: {gdf[richness_col].min()} - {gdf[richness_col].max()}")
print(f"Mean richness: {gdf[richness_col].mean():.2f} ± {gdf[richness_col].std():.2f}")

# Create spatial weights matrix based on Queen contiguity
print("\nBuilding spatial weights matrix (Queen contiguity)...")
w = Queen.from_dataframe(gdf, use_index=False)
print(f"Weights matrix: {w.n} observations, {w.pct_nonzero:.2%} non-zero")

# Compute Global Moran's I.
#
# The permutation p-value is a Monte Carlo estimate, so without a fixed seed it
# moves between runs — repeated runs of this script gave p = 0.002, 0.003 and 0.005
# for the same data, and the manuscript text and the Figure S1 caption ended up
# quoting two different draws. esda's Moran takes no seed argument, so the global
# NumPy seed is the only lever; it is sufficient, and verified reproducible here.
SEED = 42
np.random.seed(SEED)
print(f"\nComputing Global Moran's I (999 permutations, seed {SEED})...")
moran = Moran(gdf[richness_col].values, w, permutations=999)

print(f"\n{'='*60}")
print(f"GLOBAL MORAN'S I RESULTS")
print(f"{'='*60}")
print(f"Moran's I:           {moran.I:.4f}")
print(f"Expected I:          {moran.EI:.4f}")
print(f"Variance:            {moran.VI_norm:.6f}")
print(f"Z-score:             {moran.z_norm:.4f}")
print(f"P-value (normal):    {moran.p_norm:.4f}")
print(f"P-value (permutation): {moran.p_sim:.4f}")
print(f"{'='*60}")

if moran.p_sim < 0.001:
    sig_text = "p < 0.001 (highly significant)"
elif moran.p_sim < 0.01:
    sig_text = "p < 0.01 (very significant)"
elif moran.p_sim < 0.05:
    sig_text = "p < 0.05 (significant)"
else:
    sig_text = f"p = {moran.p_sim:.4f} (not significant)"

if moran.I > moran.EI:
    interpretation = f"Positive spatial autocorrelation ({sig_text})"
    interpretation += "\nNeighbouring cells tend to have similar richness values."
else:
    interpretation = f"Negative spatial autocorrelation ({sig_text})"
    interpretation += "\nNeighbouring cells tend to have dissimilar richness values."

print(f"\nInterpretation: {interpretation}")

# Save results
results_df = pd.DataFrame({
    'Statistic': ['Moran\'s I', 'Expected I', 'Variance', 'Z-score', 'P-value (permutation)', 'P-value (normal)', 'Significance'],
    'Value': [
        f"{moran.I:.4f}",
        f"{moran.EI:.4f}",
        f"{moran.VI_norm:.6f}",
        f"{moran.z_norm:.4f}",
        f"{moran.p_sim:.4f}",
        f"{moran.p_norm:.4f}",
        sig_text
    ]
})
results_df.to_csv('morans_i_results.csv', index=False)
print(f"\nResults saved to: morans_i_results.csv")

# Create visualization
print("\nCreating spatial autocorrelation map...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Species richness with spatial autocorrelation context
ax1 = axes[0]
gdf.plot(column=richness_col,
         cmap='YlOrRd',
         edgecolor='black',
         linewidth=0.5,
         legend=True,
         legend_kwds={'label': 'Species Richness', 'orientation': 'vertical'},
         ax=ax1)
ax1.set_title(f'(A) Species Richness Distribution\nMoran\'s I = {moran.I:.3f}, p = {moran.p_sim:.3f}',
              fontsize=11, fontweight='bold')
ax1.set_xlabel('Longitude (°E)', fontsize=9)
ax1.set_ylabel('Latitude (°S)', fontsize=9)
ax1.tick_params(labelsize=8)
ax1.grid(True, alpha=0.3)

# Add north arrow
arrow_x = ax1.get_xlim()[0] + 0.05 * (ax1.get_xlim()[1] - ax1.get_xlim()[0])
arrow_y = ax1.get_ylim()[1] - 0.1 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
ax1.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 0.02),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=10, ha='center')

# Panel B: Local spatial autocorrelation (LISA - Moran scatter plot concept)
ax2 = axes[1]

# Compute spatially lagged richness.
#
# Moran() sets w.transform = 'r' in place, so w.sparse.dot(y) is already the mean
# of each cell's neighbours. This previously divided that by the cardinality a
# second time, which shrank the lag values and produced a scatter whose slope was
# negative while Moran's I was positive. With the lag computed correctly the OLS
# slope equals Moran's I, as it should.
gdf['lag_richness'] = w.sparse.dot(gdf[richness_col].values)

# Scatter plot
scatter = ax2.scatter(gdf[richness_col], gdf['lag_richness'],
                     c=gdf[richness_col], cmap='YlOrRd',
                     edgecolors='black', linewidth=0.5, s=80, alpha=0.7)

# Add regression line
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(gdf[richness_col], gdf['lag_richness'])
x_line = np.linspace(gdf[richness_col].min(), gdf[richness_col].max(), 100)
y_line = slope * x_line + intercept
ax2.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7, label=f'Slope = {slope:.3f}')

# Add mean lines for quadrant reference
ax2.axvline(gdf[richness_col].mean(), color='gray', linestyle=':', alpha=0.5)
ax2.axhline(gdf['lag_richness'].mean(), color='gray', linestyle=':', alpha=0.5)

ax2.set_xlabel('Species Richness (observed)', fontsize=9)
ax2.set_ylabel('Spatial Lag (neighbors\' mean richness)', fontsize=9)
ax2.set_title(f'(B) Moran Scatter Plot\nR² = {r_value**2:.3f}',
              fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=8)

# Add colorbar for Panel B
cbar = plt.colorbar(scatter, ax=ax2, orientation='vertical', pad=0.02, fraction=0.046)
cbar.set_label('Richness', fontsize=8)
cbar.ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('figures/morans_i_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/morans_i_analysis.png', dpi=300, bbox_inches='tight')
print("Map saved to: figures/morans_i_analysis.pdf and .png")

print("\n✓ Analysis complete!")
print(f"\nFor manuscript:")
print(f"  Moran's I = {moran.I:.3f}, Z = {moran.z_norm:.2f}, p = {moran.p_sim:.3f}")
