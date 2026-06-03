"""Directed warehouse routing graph.

Layout conventions (same as the production system)
---------------------------------------------------
Tile ID    : "T{x:03d}_{y:03d}"
Odd x      : one-way aisle — edges only in the y-ascending direction (south → north)
Even x     : bidirectional aisle
Corridors  : bidirectional edges at global y_min / y_max between adjacent aisles
Depot      : "__DEPOT__" — bidirectionally connected to every aisle bottom tile

Self-contained — no imports from the main repository.
"""
from __future__ import annotations

import heapq
from collections import defaultdict

import pandas as pd

UNIT_DIST = 2.0      # metres per grid unit
DEPOT = "__DEPOT__"


class WarehouseGraph:
    """Directed graph built from location (x_coord, y_coord) pairs.

    Distances are computed on-demand with Dijkstra and cached per source tile,
    so the first query from a given tile is O(E log V); subsequent ones are O(1).
    """

    def __init__(self, locations_df: pd.DataFrame) -> None:
        self._loc_to_tile: dict[str, str] = {}
        self._adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self._oneway_xs: set[int] = set()
        self._cache: dict[str, dict[str, float]] = {}
        self._build(locations_df)

    # ------------------------------------------------------------------
    def _build(self, locs: pd.DataFrame) -> None:
        xs = sorted(locs["x_coord"].unique().astype(int))
        ys_global = sorted(locs["y_coord"].unique().astype(int))
        y_min, y_max = ys_global[0], ys_global[-1]

        # Map every location_id to its tile ID
        for row in locs.itertuples(index=False):
            tile = f"T{int(row.x_coord):03d}_{int(row.y_coord):03d}"
            self._loc_to_tile[row.location_id] = tile

        # Build intra-aisle edges (along the y axis)
        x_to_ys: dict[int, list[int]] = {}
        for x in xs:
            ys = sorted(locs[locs["x_coord"] == x]["y_coord"].unique().astype(int))
            x_to_ys[x] = ys
            one_way = (x % 2 == 1)   # odd columns are one-way (northbound only)
            if one_way:
                self._oneway_xs.add(x)
            for i in range(len(ys) - 1):
                t_a = f"T{x:03d}_{ys[i]:03d}"
                t_b = f"T{x:03d}_{ys[i + 1]:03d}"
                d = (ys[i + 1] - ys[i]) * UNIT_DIST
                self._adj[t_a].append((t_b, d))
                if not one_way:
                    self._adj[t_b].append((t_a, d))

        # Corridor edges between adjacent aisles at global top and bottom rows
        for i in range(len(xs) - 1):
            x_a, x_b = xs[i], xs[i + 1]
            for y_target in (y_min, y_max):
                tiles = []
                for x in (x_a, x_b):
                    ys = x_to_ys.get(x, [])
                    if ys:
                        closest_y = min(ys, key=lambda y: abs(y - y_target))
                        tiles.append(f"T{x:03d}_{closest_y:03d}")
                if len(tiles) == 2:
                    d = abs(x_b - x_a) * UNIT_DIST
                    self._adj[tiles[0]].append((tiles[1], d))
                    self._adj[tiles[1]].append((tiles[0], d))

        # Connect depot bidirectionally to every aisle's bottom tile
        self._adj[DEPOT]   # ensure the key exists even if no aisles are present
        for x, ys in x_to_ys.items():
            t_bottom = f"T{x:03d}_{ys[0]:03d}"
            self._adj[DEPOT].append((t_bottom, UNIT_DIST))
            self._adj[t_bottom].append((DEPOT, UNIT_DIST))

    # ------------------------------------------------------------------
    def _dijkstra_from(self, src: str) -> dict[str, float]:
        """Shortest distances from *src* to all reachable nodes (cached)."""
        if src in self._cache:
            return self._cache[src]
        dist: dict[str, float] = {src: 0.0}
        heap: list[tuple[float, str]] = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            for v, w in self._adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = dist
        return dist

    def distance(self, a: str, b: str) -> float:
        """Shortest-path distance in metres between two tiles (or DEPOT)."""
        return self._dijkstra_from(a).get(b, float("inf"))

    def loc_to_tile(self, location_id: str) -> str:
        """Return the tile for *location_id*; unmapped locations fall back to DEPOT."""
        return self._loc_to_tile.get(location_id, DEPOT)

    @property
    def start_tile(self) -> str:
        return DEPOT

    @property
    def end_tile(self) -> str:
        return DEPOT

    def oneway_xs(self) -> set[int]:
        """Return the set of x-coordinates that are one-way aisles."""
        return {int(x) for x in self._oneway_xs}

    def tile_x(self, tile: str) -> int | None:
        """Extract the x-coordinate from a tile ID; returns None for DEPOT."""
        if tile == DEPOT:
            return None
        try:
            return int(tile[1:4])
        except (ValueError, IndexError):
            return None


def depot_distances(locations: pd.DataFrame) -> dict[str, float]:
    """Shortest-path distance (metres) from DEPOT to each location.

    Used by data_gen to seed the ``time_to_start_point`` column in
    ``loc_features``.  Builds the full graph once, so it's O(V log V + E).
    """
    g = WarehouseGraph(locations)
    return {
        loc_id: g.distance(DEPOT, g.loc_to_tile(loc_id))
        for loc_id in locations["location_id"]
    }
