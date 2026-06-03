"""TSP solvers for warehouse pick-route optimisation.

Available solvers (pass ``solver=`` to ``route_all_pickruns``):

  "nn"        — nearest-neighbor greedy baseline  (default)
  "2opt"      — NN + 2-opt local search; ~5–15% shorter routes
  "or_opt"    — NN + or-opt; relocates chains of 1–2 tiles to better gaps
  "sa"        — simulated annealing; escapes local optima at the cost of speed
  "aisle_nn"  — S-shape sweep sorted by aisle column, then NN within each aisle
  "bucketed"  — nearest-neighbor bucketing + brute-force within each bucket
  "mst"       — Christofides-like heuristic (MST + greedy matching); needs scipy
  "aco"       — Ant Colony Optimisation; pheromone-guided probabilistic search

Pick-ordering constraints (same in every solver):

  Segment 1: heavy items   (weight ≥ 500 kg) — picked first
  Segment 2: normal items
  Segment 3: fragile items (weight ≤  50 kg) — picked last

Each segment is routed independently, so the ordering guarantee is hard.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import permutations
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
# Or-opt improvement
# ---------------------------------------------------------------------------

def or_opt_improve(
    segment: list[str],
    entry: str,
    graph: WarehouseGraph,
    max_chain: int = 2,
) -> tuple[list[str], float]:
    """Or-opt local search: relocate runs of 1..max_chain tiles to a better gap.

    Unlike 2-opt (which reverses sub-sequences), or-opt lifts a consecutive
    chain and re-inserts it at every other position.  Catches improvements
    2-opt misses when the optimal move is a simple shift rather than a reversal.
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
        for chain in range(1, min(max_chain + 1, len(best))):
            for i in range(len(best) - chain + 1):
                chunk = best[i : i + chain]
                rest  = best[:i] + best[i + chain :]
                for j in range(len(rest) + 1):
                    candidate = rest[:j] + chunk + rest[j:]
                    d = full_cost(candidate)
                    if d < best_dist - 1e-9:
                        best, best_dist = candidate, d
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best, best_dist


# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------

def simulated_annealing(
    tiles: list[str],
    start: str,
    graph: WarehouseGraph,
    n_iter: int = 4_000,
    T_start: float = 100.0,
    alpha: float = 0.995,
) -> tuple[list[str], float]:
    """Simulated annealing TSP.

    Starts from a nearest-neighbor seed and applies random 2-opt moves,
    accepting worse solutions with probability exp(-Δ/T).  The temperature T
    decays geometrically (×alpha each step).  Higher n_iter or T_start allows
    more exploration at the cost of runtime.
    """
    if not tiles:
        return [], 0.0

    route, _ = nearest_neighbor(tiles, start, graph)

    def cost(r: list[str]) -> float:
        return graph.distance(start, r[0]) + _segment_distance(r, graph)

    best     = list(route)
    best_c   = cost(best)
    cur      = list(route)
    cur_c    = best_c
    T        = T_start
    rng      = np.random.default_rng(0)

    for _ in range(n_iter):
        if len(cur) < 2:
            break
        i, j = sorted(rng.integers(0, len(cur), size=2))
        if i == j:
            continue
        cand  = cur[:i] + cur[i : j + 1][::-1] + cur[j + 1 :]
        cand_c = cost(cand)
        delta  = cand_c - cur_c
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            cur, cur_c = cand, cand_c
            if cur_c < best_c:
                best, best_c = list(cur), cur_c
        T *= alpha

    return best, best_c


# ---------------------------------------------------------------------------
# Aisle-sorted nearest-neighbor  (S-shape sweep)
# ---------------------------------------------------------------------------

def aisle_sorted_nn(
    tiles: list[str],
    start: str,
    graph: WarehouseGraph,
) -> tuple[list[str], float]:
    """S-shape sweep: visit aisles left → right, alternating direction each aisle.

    Inspired by the directed-TSP approach used in the production system.
    Groups pick tiles by their x-column (aisle), sorts aisles ascending, then
    for each aisle visits tiles in ascending-y order (south→north) or descending
    alternately — except for one-way aisles, which are always south→north.
    """
    if not tiles:
        return [], 0.0

    by_x: dict[int, list[str]] = defaultdict(list)
    for t in tiles:
        x = graph.tile_x(t)
        by_x[x if x is not None else -999].append(t)

    oneway = graph.oneway_xs()
    route:  list[str] = []
    total:  float     = 0.0
    cur = start

    for i, x in enumerate(sorted(by_x)):
        aisle_tiles = sorted(by_x[x], key=lambda t: int(t.split("_")[1]))
        if x not in oneway and i % 2 == 1:
            aisle_tiles = aisle_tiles[::-1]   # reverse on even-indexed bidirectional aisles
        for t in aisle_tiles:
            total += graph.distance(cur, t)
            route.append(t)
            cur = t

    return route, total


# ---------------------------------------------------------------------------
# Bucketed nearest-neighbor with brute-force within each bucket
# ---------------------------------------------------------------------------

def bucketed_brute(
    tiles: list[str],
    start: str,
    graph: WarehouseGraph,
    bucket_size: int = 6,
) -> tuple[list[str], float]:
    """Nearest-neighbor bucketing + brute-force within each bucket.

    1. From the current position, greedily collect the ``bucket_size`` nearest
       unvisited tiles to form a sub-problem.
    2. Enumerate all permutations of that bucket and keep the best.
    3. Repeat from the end of the solved bucket until all tiles are visited.

    Larger bucket_size → better quality, exponentially slower.
    Capped at 7 to keep runtime sane (7! = 5040 permutations).
    """
    bucket_size = min(bucket_size, 7)
    if not tiles:
        return [], 0.0

    remaining = list(tiles)
    route: list[str] = []
    total = 0.0
    cur   = start

    while remaining:
        # Pick the nearest bucket_size tiles as candidates
        bucket = sorted(remaining, key=lambda t: graph.distance(cur, t))[:bucket_size]

        best_perm: tuple[str, ...] = tuple(bucket)
        best_d = float("inf")
        for perm in permutations(bucket):
            d = graph.distance(cur, perm[0]) + _segment_distance(list(perm), graph)
            if d < best_d:
                best_d, best_perm = d, perm

        for t in best_perm:
            remaining.remove(t)
        route.extend(best_perm)
        total += best_d
        cur = best_perm[-1]

    return route, total


# ---------------------------------------------------------------------------
# MST / Christofides-like heuristic
# ---------------------------------------------------------------------------

def _hierholzer(adj: dict[int, list[int]], start: int) -> list[int]:
    """Hierholzer's Eulerian circuit on a multigraph (adjacency consumed in-place)."""
    circuit: list[int] = []
    stack   = [start]
    while stack:
        v = stack[-1]
        if adj[v]:
            u = adj[v].pop()
            try:
                adj[u].remove(v)
            except ValueError:
                pass
            stack.append(u)
        else:
            circuit.append(stack.pop())
    return circuit[::-1]


def mst_christofides(
    tiles: list[str],
    start: str,
    graph: WarehouseGraph,
) -> tuple[list[str], float]:
    """Christofides-like heuristic: MST + greedy odd-degree matching → Hamiltonian tour.

    Steps
    -----
    1. Build a distance matrix for {start} ∪ tiles.
    2. Find the Minimum Spanning Tree (requires scipy).
    3. Identify odd-degree nodes in the MST; greedily pair them with minimum-
       weight edges to make the graph Eulerian.
    4. Find an Eulerian circuit (Hierholzer's algorithm).
    5. Shortcut to a Hamiltonian path by skipping already-visited nodes.

    The result is a 1.5-approximation on metric instances (with optimal matching);
    our greedy matching is slightly weaker but much faster.

    Requires: ``pip install scipy``
    """
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree as _mst
    except ImportError:
        raise ImportError("mst solver requires scipy — pip install scipy")

    if not tiles:
        return [], 0.0
    if len(tiles) == 1:
        return list(tiles), graph.distance(start, tiles[0])

    nodes = [start] + list(dict.fromkeys(tiles))   # dedup, start at index 0
    n     = len(nodes)

    D = np.array([[graph.distance(nodes[i], nodes[j]) for j in range(n)]
                  for i in range(n)])

    # --- MST ---
    mst_arr = _mst(csr_matrix(D)).toarray()
    adj_mst: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        for j in range(n):
            if mst_arr[i, j] > 0:
                adj_mst[i].append(j)
                adj_mst[j].append(i)

    # --- odd-degree nodes ---
    odd = [i for i in range(n) if len(adj_mst[i]) % 2 == 1]

    # --- greedy min-weight matching ---
    extra: list[tuple[int, int]] = []
    remaining = list(odd)
    while len(remaining) >= 2:
        best_d, best_a, best_b = float("inf"), 0, 1
        for a in range(len(remaining)):
            for b in range(a + 1, len(remaining)):
                u, v = remaining[a], remaining[b]
                if D[u, v] < best_d:
                    best_d, best_a, best_b = D[u, v], a, b
        u, v = remaining[best_a], remaining[best_b]
        extra.append((u, v))
        # Remove higher index first to keep indices valid
        for idx in sorted((best_a, best_b), reverse=True):
            remaining.pop(idx)

    # --- multigraph adjacency ---
    multi: dict[int, list[int]] = {i: list(adj_mst[i]) for i in range(n)}
    for u, v in extra:
        multi[u].append(v)
        multi[v].append(u)

    # --- Eulerian circuit from node 0 ---
    euler = _hierholzer(multi, 0)

    # --- shortcut to Hamiltonian ---
    seen:  set[int]  = set()
    ham:   list[int] = []
    for node in euler:
        if node not in seen:
            seen.add(node)
            ham.append(node)

    # ham[0] is node 0 = start; skip it
    route = [nodes[i] for i in ham[1:]]
    for t in tiles:          # safety: add any tiles that got dropped
        if t not in set(route):
            route.append(t)

    total = (graph.distance(start, route[0]) + _segment_distance(route, graph)
             if route else 0.0)
    return route, total


# ---------------------------------------------------------------------------
# Ant Colony Optimisation
# ---------------------------------------------------------------------------

def ant_colony(
    tiles: list[str],
    start: str,
    graph: WarehouseGraph,
    n_ants: int = 15,
    n_iterations: int = 50,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.5,
) -> tuple[list[str], float]:
    """Ant Colony Optimisation (ACO) TSP.

    Ants construct tours stochastically, choosing the next tile with probability
    proportional to  τ^alpha × η^beta  where τ is pheromone intensity and η is
    the inverse distance (heuristic desirability).  After each iteration,
    pheromones evaporate by factor rho and are reinforced along edges used by
    any ant, proportional to solution quality.

    Ported from the production codebase (aco1_TSP) — Numba dependency removed,
    pure NumPy.  Several improvements are left as exercises:

      • Elitism: deposit extra pheromone on the globally best tour each iteration
        so good solutions are reinforced more aggressively.
      • Per-ant 2-opt: apply a few 2-opt passes to each ant's tour before
        computing the pheromone deposit (see ``two_opt_improve``).
      • Symmetric deposit: currently pheromone[i, j] is updated but not
        pheromone[j, i] — does making it symmetric improve convergence?
      • Parameter sweep: try alpha ∈ [0.5, 2], beta ∈ [1, 5], rho ∈ [0.1, 0.9].
    """
    if not tiles:
        return [], 0.0
    if len(tiles) == 1:
        return list(tiles), graph.distance(start, tiles[0])

    # Include start as node 0 so the distance matrix is self-contained.
    nodes = [start] + list(dict.fromkeys(tiles))
    n     = len(nodes)

    D = np.array(
        [[graph.distance(nodes[i], nodes[j]) for j in range(n)] for i in range(n)],
        dtype=np.float64,
    )
    np.fill_diagonal(D, 0.0)

    heuristic = 1.0 / (D + 1e-10)              # η: prefer short edges
    pheromone = np.full((n, n), 0.1)            # τ: uniform initial pheromone

    best_route_idx: list[int] = list(range(1, n))
    best_cost = D[0, 1] + sum(D[best_route_idx[k], best_route_idx[k + 1]]
                               for k in range(n - 2))
    rng = np.random.default_rng(0)

    for _ in range(n_iterations):
        iter_routes: list[list[int]] = []
        iter_costs:  list[float]     = []

        for _ in range(n_ants):
            unvisited = list(range(1, n))   # tiles; start is always node 0
            route     = [0]

            while unvisited:
                cur     = route[-1]
                attract = (pheromone[cur, unvisited] ** alpha
                           * heuristic[cur, unvisited] ** beta)
                total   = attract.sum()
                probs   = attract / total if total > 0 else np.ones(len(unvisited)) / len(unvisited)
                nxt     = unvisited[int(rng.choice(len(unvisited), p=probs))]
                route.append(nxt)
                unvisited.remove(nxt)

            cost = sum(D[route[k], route[k + 1]] for k in range(len(route) - 1))
            iter_routes.append(route)
            iter_costs.append(cost)

        # Track global best
        mi = int(np.argmin(iter_costs))
        if iter_costs[mi] < best_cost:
            best_cost      = iter_costs[mi]
            best_route_idx = iter_routes[mi][1:]   # drop start node (index 0)

        # Pheromone update: evaporate then deposit proportional to solution quality.
        # Note: only pheromone[i, j] is updated (not [j, i]) — is this right for
        # directed warehouse graphs?  Try making it symmetric and compare results.
        pheromone *= (1.0 - rho)
        for route, cost in zip(iter_routes, iter_costs):
            if cost == 0.0:
                continue
            deposit = 1.0 / cost
            for k in range(len(route) - 1):
                pheromone[route[k], route[k + 1]] += deposit

    route_tiles = [nodes[i] for i in best_route_idx]
    visited = set(route_tiles)
    for t in tiles:
        if t not in visited:
            route_tiles.append(t)

    total = (graph.distance(start, route_tiles[0]) + _segment_distance(route_tiles, graph)
             if route_tiles else 0.0)
    return route_tiles, total


# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------

SOLVERS = frozenset({"nn", "2opt", "or_opt", "sa", "aisle_nn", "bucketed", "mst", "aco"})


def _apply_solver(
    solver: str,
    tiles: list[str],
    entry: str,
    graph: WarehouseGraph,
    **kw,
) -> tuple[list[str], float]:
    """Dispatch to the named solver; return (ordered_tiles, distance_from_entry)."""
    if solver == "nn":
        return nearest_neighbor(tiles, entry, graph)
    if solver == "2opt":
        route, _ = nearest_neighbor(tiles, entry, graph)
        return two_opt_improve(route, entry, graph)
    if solver == "or_opt":
        route, _ = nearest_neighbor(tiles, entry, graph)
        return or_opt_improve(route, entry, graph, **kw)
    if solver == "sa":
        return simulated_annealing(tiles, entry, graph, **kw)
    if solver == "aisle_nn":
        return aisle_sorted_nn(tiles, entry, graph)
    if solver == "bucketed":
        return bucketed_brute(tiles, entry, graph, **kw)
    if solver == "mst":
        return mst_christofides(tiles, entry, graph)
    if solver == "aco":
        return ant_colony(tiles, entry, graph, **kw)
    raise ValueError(f"Unknown solver {solver!r}. Choose from: {sorted(SOLVERS)}")


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
    solver: str = "nn",
    **solver_kwargs,
) -> RouteResult:
    """Route one pickrun with heavy → normal → fragile segment ordering.

    Parameters
    ----------
    pickrun_items  : DataFrame with at least columns [item, location_id]
    graph          : WarehouseGraph
    item_weight    : {item: weight_kg} lookup
    solver         : one of "nn", "2opt", "or_opt", "sa", "aisle_nn",
                     "bucketed", "mst"  (see module docstring)
    **solver_kwargs: forwarded to the solver (e.g. bucket_size=5, n_iter=6000)
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
        seg_route, seg_dist = _apply_solver(solver, segment, cur, graph, **solver_kwargs)
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
    solver: str = "nn",
    **solver_kwargs,
) -> list[RouteResult]:
    """Route every pickrun in ``transactions``.

    Parameters
    ----------
    transactions  : DataFrame with columns [pickrun_no, item, location_id, ...]
    graph         : WarehouseGraph
    items_df      : items DataFrame with at least [item, weight] columns
    max_pickruns  : cap on pickruns routed (useful for quick iteration)
    solver        : routing algorithm — see module docstring for options
    **solver_kwargs: forwarded to the solver (e.g. bucket_size=5)

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
        rr = route_pickrun(rows, graph, item_weight, solver=solver, **solver_kwargs)
        release = release_times.get(str(pr_no), 0.0)
        results.append(rr._replace(release_s=release))

    return results
