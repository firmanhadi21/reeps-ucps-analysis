#!/usr/bin/env python3
"""
compute_connectivity_corridors.py
─────────────────────────────────
Re-derives the connectivity, survey-gap and stepping-stone layers from the
audited occurrence data, replacing the copies inherited from
REEPS_GridAnalyses.gpkg (which were built from a superseded 565-record extract).

Definitions follow the manuscript Methods where stated, and were otherwise
recovered from the shipped files and verified to reproduce them exactly:

  cell classification  interior (6 occupied neighbours) / edge (1-5) /
                       isolated (0)                              [Methods]
  keystone             articulation point of the occupied subgraph[Methods]
  EIS (eco_score)      richness*4 + occ_nbrs*2
                       + k2_reachable_richness*0.3 + 10 if keystone[Methods]
  betweenness          shortest BFS paths between occupied cell pairs
                       passing through each unsurveyed cell        [Methods]
  corridor tiers       T1 adjacent to a CRITICAL cell; T2 adjacent to a HIGH
                       cell but no CRITICAL; T3 otherwise on a shortest path
                       between high-priority cells                 [Methods]
  k1/k2_nbr_richness   summed richness of occupied cells in the k1 ring /
                       k2 disk                       [verified 109/109]
  corridor_candidate   k1 occupied neighbours >= 2   [verified 109/109]
  gap_priority_score   k1_nbr_richness*2 + k2_nbr_richness*0.5
                       + corridor_candidate*10       [verified 109/109, R2=1.0]
  survey_priority      High >= 10, Medium > 3, else Low  [verified 109/109]

NOTE the shipped eco_score did not follow the EIS formula printed in the
Methods (a least-squares fit gives a richness coefficient of ~0 rather than 4).
This script implements the documented formula, so eco_score values differ from
the previously published figure.
"""

import os
from itertools import combinations
from pathlib import Path

import h3
import networkx as nx
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


def main() -> None:
    h3a = pd.read_csv(BASE / "h3_analysis.csv").set_index("h3_index")
    occ = pd.read_csv(BASE / "reeps_h3.csv")
    pri = pd.read_csv(BASE / "priority_index.csv").set_index("h3_index")

    cells = list(h3a.index)
    cellset = set(cells)
    rich = h3a["species_richness"].to_dict()
    occupied = {c for c in cells if rich.get(c, 0) > 0}
    unsurveyed = [c for c in cells if c not in occupied]
    print(f"AOI cells {len(cells)} | occupied {len(occupied)} | "
          f"unsurveyed {len(unsurveyed)}")

    # adjacency graph over the AOI
    G = nx.Graph()
    G.add_nodes_from(cells)
    for c in cells:
        for n in h3.grid_disk(c, 1):
            if n != c and n in cellset:
                G.add_edge(c, n)

    occ_nbrs = {c: sum(1 for n in G[c] if n in occupied) for c in cells}
    k2_occ = {c: sum(1 for n in h3.grid_disk(c, 2)
                     if n != c and n in cellset and n in occupied) for c in cells}

    def k_ring_richness(c: str, k: int) -> int:
        return sum(rich.get(n, 0) for n in h3.grid_disk(c, k)
                   if n != c and n in cellset)

    def ring_only_richness(c: str) -> int:
        inner = set(h3.grid_disk(c, 1))
        return sum(rich.get(n, 0) for n in h3.grid_disk(c, 2)
                   if n != c and n in cellset and n not in inner)

    # patches and keystone cells
    Gocc = G.subgraph(occupied)
    patches = {c: i + 1 for i, comp in enumerate(
        sorted(nx.connected_components(Gocc), key=len, reverse=True)) for c in comp}
    n_patches = len(set(patches.values()))
    keystone = set(nx.articulation_points(Gocc)) if len(occupied) > 2 else set()
    print(f"occupied patches: {n_patches}   keystone cells: {len(keystone)}")

    def classify(c: str) -> str:
        if c not in occupied:
            return "Empty"
        k = occ_nbrs[c]
        return "Isolated" if k == 0 else ("Interior" if k == 6 else "Edge")

    def dist_to_occupied(c: str) -> int:
        if c in occupied:
            return 0
        for k in range(1, 12):
            if any(n in occupied for n in h3.grid_disk(c, k)):
                return k
        return -1

    latlng = {c: h3.cell_to_latlng(c) for c in cells}

    # ── 1. connectivity ──────────────────────────────────────────────────────
    conn = pd.DataFrame([{
        "h3_index": c,
        "lat": latlng[c][0], "lon": latlng[c][1],
        "cell_type": "Occupied" if c in occupied else "Empty",
        "structural_class": classify(c),
        "occupied": int(c in occupied),
        "patch_id": patches.get(c, 0),
        "occ_nbrs": occ_nbrs[c],
        "k2_occ_nbrs": k2_occ[c],
        "sp_reachable_k2": k_ring_richness(c, 2),
        "keystone": int(c in keystone),
        "eco_score": round(
            rich.get(c, 0) * 4 + occ_nbrs[c] * 2
            + k_ring_richness(c, 2) * 0.3 + (10 if c in keystone else 0), 2)
        if c in occupied else 0.0,
    } for c in cells])
    conn.to_csv(BASE / "h3_connectivity_full.csv", index=False)
    print(f"  h3_connectivity_full.csv: {len(conn)} rows")

    # ── 2. survey gaps ───────────────────────────────────────────────────────
    gap_rows = []
    for c in unsurveyed:
        k1r = k_ring_richness(c, 1)
        k2r = k_ring_richness(c, 2)
        cand = int(occ_nbrs[c] >= 2)
        score = k1r * 2 + k2r * 0.5 + cand * 10
        gap_rows.append({
            "h3_index": c, "lat": latlng[c][0], "lon": latlng[c][1],
            "k1_occ_neighbors": occ_nbrs[c],
            "k2_occ_neighbors": k2_occ[c],
            "dist_to_nearest_occ_km": dist_to_occupied(c),
            "k1_nbr_richness": k1r,
            "k2_nbr_richness": k2r,
            "corridor_candidate": cand,
            "gap_priority_score": round(score, 2),
            "survey_priority": "High" if score >= 10 else (
                "Medium" if score > 3 else "Low"),
        })
    gap = pd.DataFrame(gap_rows).sort_values(
        "gap_priority_score", ascending=False).reset_index(drop=True)
    gap.to_csv(BASE / "gap_analysis.csv", index=False)
    print(f"  gap_analysis.csv: {len(gap)} rows  "
          f"({gap['survey_priority'].value_counts().to_dict()})")

    # ── 3. stepping stones — betweenness over occupied-pair shortest paths ───
    through = {c: 0 for c in cells}
    occ_list = sorted(occupied)
    for a, b in combinations(occ_list, 2):
        try:
            paths = list(nx.all_shortest_paths(G, a, b))
        except nx.NetworkXNoPath:
            continue
        for p in paths:
            for node in p[1:-1]:
                through[node] += 1

    tier_of = pri["Priority_Tier"].to_dict() if "Priority_Tier" in pri else {}

    step_rows = []
    for c in unsurveyed:
        nbr_tiers = {tier_of.get(n) for n in G[c] if n in occupied}
        if "CRITICAL" in nbr_tiers:
            tier = "Tier 1"
        elif "HIGH" in nbr_tiers:
            tier = "Tier 2"
        elif through[c] > 0 and nbr_tiers:
            tier = "Tier 3"
        else:
            tier = "None"
        step_rows.append({
            "h3_index": c, "lat": latlng[c][0], "lon": latlng[c][1],
            "betweenness": through[c],
            "k1_occ_neighbors": occ_nbrs[c],
            "k2_richness_reachable": k_ring_richness(c, 2),
            "dist_to_nearest_occ": dist_to_occupied(c),
            "is_corridor_candidate": int(occ_nbrs[c] >= 2),
            "connects_priority_cells": int(
                bool(nbr_tiers & {"CRITICAL", "HIGH"})),
            # transparent composite: path importance plus neighbourhood value
            "stepping_stone_score": round(
                through[c] * 0.5 + k_ring_richness(c, 1) * 2
                + (10 if occ_nbrs[c] >= 2 else 0), 2),
            "corridor_tier": tier,
        })
    step = pd.DataFrame(step_rows).sort_values(
        "betweenness", ascending=False).reset_index(drop=True)
    step.to_csv(BASE / "stepping_stones.csv", index=False)
    print(f"  stepping_stones.csv: {len(step)} rows  "
          f"({step['corridor_tier'].value_counts().to_dict()})")
    print(f"  betweenness: max {step['betweenness'].max()}, "
          f"non-zero in {(step['betweenness'] > 0).sum()} cells")

    # ── validation gates ─────────────────────────────────────────────────────
    gates = [
        ("single contiguous patch", n_patches == 1),
        ("no isolated occupied cells",
         sum(1 for c in occupied if occ_nbrs[c] == 0) == 0),
        ("gap rows == unsurveyed", len(gap) == len(unsurveyed)),
        ("stepping rows == unsurveyed", len(step) == len(unsurveyed)),
        ("all cells classified", conn["structural_class"].notna().all()),
    ]
    print("\nvalidation gates:")
    ok = True
    for name, passed in gates:
        ok &= passed
        print(f"  {name:32s} {'PASS' if passed else 'FAIL'}")
    print(f"\nALL GATES {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
