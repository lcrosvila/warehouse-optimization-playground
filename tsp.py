"""TSP solvers for warehouse pick-route optimisation.

Two solvers are provided as a starting point:

  nearest_neighbor  — greedy O(n²) heuristic; fast baseline
  two_opt_improve   — 2-opt local search on top of NN; ~5–15% shorter routes

The ``route_pickrun`` function applies pick-ordering constraints that mirror
real ergonomic and damage-prevention rules:

  Segment 1: heavy items   (weight ≥ 500 kg) — picked first
  Segment 2: normal items
  Segment 3: fragile items (weight ≤  50 kg) — picked last

Each segment is routed independently so the ordering guarantee is hard.
2-opt improvement is applied within each segment.

Your task
---------
  • Replace or augment the TSP solver to reduce average route distance.
  • Ideas: or-opt (move single tiles without reversing), 3-opt, simulated
    annealing, zone-aware routing, or a learned heuristic.
  • ``route_all_pickruns`` accepts an ``improve`` flag you can swap for your
    own solver flag once you add a new function.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from graph import WarehouseGraph, DEPOT

SPEED_M_S   = 1.5   # picker walking speed in metres / second
PICK_TIME_S = 4.0   # time to pick one item in seconds

HEAVY_KG   = 500    # weight threshold for "heavy" pick class
FRAGILE_KG =  50    # weight threshold for "fragile" pick class


class RouteResult(NamedTuple):
    route_tiles:    list[str]  # ordered tile sequence: start → picks → end
    total_dist_m:   float      # total walking distance in metres
    total_time_s:   float      # estimated time (travel + pick stops), no contention
    n_picks:        int        # number of pick stops
    ordering_fixes: int        # items out of heavy→normal→fragile order in the original tx
    release_s:      float = 0.0  # seconds from day origin (derived from timestamps)


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _segment_distance(tiles: list[str], graph: WarehouseGraph) -> float:
    """Sum of consecutive-edge distances within a tile sequence (no depot legs)."""
    if len(tiles) <= 1:
        return 0.0
    return sum(graph.distance(tiles[i], tiles[i + 1]) for i in range(len(tiles) - 1))


# ---------------------------------------------------------------------------
# Nearest-neighbor TSP
# ---------------------------------------------------------------------------

def nearest_neighbor(
    tiles: list[str],
    start: str,
    graph: WarehouseGraph,
) -> tuple[list[str], float]:
    """Greedy nearest-neighbor TSP starting from ``start``.

    Returns ``(ordered_tiles, total_distance_m)`` where ``total_distance_m``
    includes the approach from ``start`` to the first chosen tile.
    Does *not* include a return leg — the caller chains segments.
    """
    remaining = list(tiles)
    route: list[str] = []
    total = 0.0
    cur = start
    while remaining:
        best_d, best_i = float("inf"), 0
        for i, t in enumerate(remaining):
            d = graph.distance(cur, t)
            if d < best_d:
                best_d, best_i = d, i
        nxt = remaining.pop(best_i)
        total += best_d
        route.append(nxt)
        cur = nxt
    return route, total


# ---------------------------------------------------------------------------
# 2-opt improvement
# ---------------------------------------------------------------------------

def two_opt_improve(
    segment: list[str],
    entry: str,
    graph: WarehouseGraph,
) -> tuple[list[str], float]:
    """2-opt local search within a pick segment.

    Repeatedly tries reversing sub-sequences of ``segment`` until no swap
    reduces the total cost.  The cost includes the edge from ``entry`` to
    ``segment[0]`` so that swaps affecting the first tile are evaluated
    correctly.

    Parameters
    ----------
    segment : ordered list of tiles for this pick class (no depot)
    entry   : the tile (or DEPOT) the picker arrives from before this segment
    graph   : routing graph

    Returns
    -------
    (improved_segment, total_distance_from_entry_to_last_tile)
    """
    if len(segment) <= 1:
        d = graph.distance(entry, segment[0]) if segment else 0.0
        return segment, d

    def full_cost(tiles: list[str]) -> float:
        return graph.distance(entry, tiles[0]) + _segment_distance(tiles, graph)

    best = list(segment)
    best_dist = full_cost(best)
    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                d = full_cost(candidate)
                if d < best_dist - 1e-9:
                    best, best_dist = candidate, d
                    improved = True
                    break
            if improved:
                break
    return best, best_dist


# ---------------------------------------------------------------------------
# Item classification
# ---------------------------------------------------------------------------

def _classify(weight: float) -> str:
    if weight >= HEAVY_KG:
        return "heavy"
    if weight <= FRAGILE_KG:
        return "fragile"
    return "normal"


# ---------------------------------------------------------------------------
# Per-pickrun routing
# ---------------------------------------------------------------------------

def route_pickrun(
    pickrun_items: pd.DataFrame,
    graph: WarehouseGraph,
    item_weight: dict[str, float],
    improve: bool = False,
) -> RouteResult:
    """Route one pickrun with heavy → normal → fragile segment ordering.

    Parameters
    ----------
    pickrun_items : DataFrame with at least columns [item, location_id]
    graph         : WarehouseGraph
    item_weight   : {item: weight_kg} lookup
    improve       : apply 2-opt within each segment (slower, shorter routes)
    """
    heavy:   list[str] = []
    normal:  list[str] = []
    fragile: list[str] = []
    orig_classes: list[str] = []

    for row in pickrun_items.itertuples(index=False):
        cls = _classify(item_weight.get(row.item, 0.0))
        orig_classes.append(cls)
        tile = graph.loc_to_tile(row.location_id)
        if cls == "heavy":
            heavy.append(tile)
        elif cls == "fragile":
            fragile.append(tile)
        else:
            normal.append(tile)

    # Count ordering violations in the original transaction sequence.
    _rank = {"heavy": 0, "normal": 1, "fragile": 2}
    ordering_fixes = sum(
        1 for i in range(1, len(orig_classes))
        if _rank[orig_classes[i]] < _rank[orig_classes[i - 1]]
    )

    full_route = [graph.start_tile]
    total_dist = 0.0
    cur = graph.start_tile

    for segment in (heavy, normal, fragile):
        if not segment:
            continue
        # Always seed with NN — 2-opt needs a good starting solution
        seg_route, seg_dist = nearest_neighbor(segment, cur, graph)
        if improve and len(seg_route) > 1:
            seg_route, seg_dist = two_opt_improve(seg_route, cur, graph)
        total_dist += seg_dist
        full_route.extend(seg_route)
        cur = seg_route[-1]

    return_dist = graph.distance(cur, graph.end_tile)
    total_dist += return_dist
    full_route.append(graph.end_tile)

    n_picks = len(full_route) - 2   # exclude start and end depot
    total_time_s = total_dist / SPEED_M_S + n_picks * PICK_TIME_S
    return RouteResult(full_route, total_dist, total_time_s, n_picks, ordering_fixes)


# ---------------------------------------------------------------------------
# Release times from transaction timestamps
# ---------------------------------------------------------------------------

def _release_times(transactions: pd.DataFrame) -> dict[str, float]:
    """Per-pickrun release time (seconds from day origin) from timestamps.

    Returns an empty dict when no valid timestamps are present; the DES
    then releases all pickruns simultaneously at t=0.
    """
    if "timestamp" not in transactions.columns:
        return {}
    ts = pd.to_datetime(transactions["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        return {}
    pr_min = transactions.assign(_ts=ts).groupby("pickrun_no")["_ts"].min()
    day_origin = pr_min.min()
    if pd.isna(day_origin):
        return {}
    return {
        pr: max(0.0, (t - day_origin).total_seconds())
        for pr, t in pr_min.items()
        if pd.notna(t)
    }


# ---------------------------------------------------------------------------
# Batch routing
# ---------------------------------------------------------------------------

def route_all_pickruns(
    transactions: pd.DataFrame,
    graph: WarehouseGraph,
    items_df: pd.DataFrame,
    max_pickruns: int | None = None,
    improve: bool = False,
) -> list[RouteResult]:
    """Route every pickrun in ``transactions``.

    Parameters
    ----------
    transactions : DataFrame with columns [pickrun_no, item, location_id, ...]
    graph        : WarehouseGraph
    items_df     : items DataFrame with at least [item, weight] columns
    max_pickruns : cap on pickruns routed (useful for quick iteration)
    improve      : apply 2-opt within each pick segment

    Returns
    -------
    List of RouteResult, one per pickrun (in encounter order).
    """
    item_weight   = dict(zip(items_df["item"], items_df["weight"]))
    release_times = _release_times(transactions)
    results: list[RouteResult] = []

    for pr_no, grp in transactions.groupby("pickrun_no", sort=False):
        if max_pickruns is not None and len(results) >= max_pickruns:
            break
        rows = grp[["item", "location_id"]].dropna()
        if rows.empty:
            continue
        rr = route_pickrun(rows, graph, item_weight, improve=improve)
        release = release_times.get(str(pr_no), 0.0)
        results.append(rr._replace(release_s=release))

    return results
