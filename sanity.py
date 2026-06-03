"""sanity.py — Independent constraint checker for proposed pick routes.

Verifies hard ordering constraints and flags route-quality issues after the
TSP solver runs.  Nothing here modifies any route; it only inspects and reports.

Usage
-----
    from sanity import validate_routes
    report = validate_routes(routes, ds.transactions, graph,
                             ds.items, ds.locations, cold_last=True)
    print(report.summary())

Hard checks  (affect ``report.passed``)
----------------------------------------
1. Start / end at depot      — every route must begin and end at ``__DEPOT__``
2. Route completeness        — every pick location in the pickrun must appear
                               in the route at least once
3. Weight segment ordering   — all heavy picks before any normal; all normal
                               before any fragile (within each temperature group)
4. Cold-chain ordering       — (only when ``cold_last=True``) all AMBIENT picks
                               before CHILLER; all CHILLER before FREEZER

Quality notes  (shown in the summary but do NOT affect ``passed``)
-------------------------------------------------------------------
5. One-way aisle order       — consecutive tile stops in the same one-way aisle
                               should be in ascending y order; if not, the picker
                               must detour via the cross-aisle corridor (valid
                               but needlessly long).  Reported as a count of
                               such suboptimal consecutive pairs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from graph import WarehouseGraph, DEPOT
from tsp import RouteResult, COLD_ZONES, HEAVY_KG, FRAGILE_KG

# Re-declare locally so this module stays independent of tsp.py internals.
_WEIGHT_CLS  = ("heavy", "normal", "fragile")
_WEIGHT_RANK = {w: i for i, w in enumerate(_WEIGHT_CLS)}
_ZONE_RANK   = {z: i for i, z in enumerate(COLD_ZONES)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tile_xy(tile: str) -> tuple[int | None, int | None]:
    """Return (x, y) parsed from a tile ID like 'T003_007', or (None, None)."""
    if tile == DEPOT:
        return None, None
    try:
        return int(tile[1:4]), int(tile[5:8])
    except (ValueError, IndexError):
        return None, None


def _weight_class(weight: float) -> str:
    if weight >= HEAVY_KG:
        return "heavy"
    if weight <= FRAGILE_KG:
        return "fragile"
    return "normal"


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Collects the results of all constraint and quality checks."""
    n_routes:              int       = 0
    depot_failures:        list[str] = field(default_factory=list)
    completeness_failures: list[str] = field(default_factory=list)
    ordering_violations:   list[str] = field(default_factory=list)
    oneway_suboptimal:     int       = 0   # quality note only

    @property
    def passed(self) -> bool:
        """True iff every hard constraint check passed for every route."""
        return not (self.depot_failures
                    or self.completeness_failures
                    or self.ordering_violations)

    def summary(self) -> str:
        def _row(label: str, issues: list[str]) -> list[str]:
            tag = "ok" if not issues else f"FAIL  ({len(issues)} violation(s))"
            rows = [f"  {label:<30} {tag}"]
            for msg in issues[:3]:
                rows.append(f"      {msg}")
            if len(issues) > 3:
                rows.append(f"      … and {len(issues) - 3} more")
            return rows

        lines = [f"  Routes checked                 : {self.n_routes}"]
        lines += _row("Start / end at depot",  self.depot_failures)
        lines += _row("Route completeness",    self.completeness_failures)
        lines += _row("Segment ordering",      self.ordering_violations)
        if self.oneway_suboptimal:
            lines.append(
                f"  One-way aisle order (quality)  "
                f"{self.oneway_suboptimal} suboptimal consecutive pair(s)  "
                f"[not a hard failure — picker detours via corridor]"
            )
        else:
            lines.append(f"  One-way aisle order (quality)  ok")
        lines.append("")
        lines.append(f"  Result : {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate_routes(
    route_results: list[RouteResult],
    transactions: pd.DataFrame,
    graph: WarehouseGraph,
    items_df: pd.DataFrame | None = None,
    locations_df: pd.DataFrame | None = None,
    cold_last: bool = False,
) -> ValidationReport:
    """Run all checks against ``route_results`` and return a report.

    Parameters
    ----------
    route_results
        Output of ``tsp.route_all_pickruns()``.
    transactions
        Raw transaction table (pickrun_no, item, location_id, …).
        Must be in the same pickrun encounter order as ``route_results``.
    graph
        Routing graph used when the routes were generated.
    items_df
        Items table with at least [item, weight].  Required for the ordering
        check; if omitted the ordering check is skipped.
    locations_df
        Locations table with at least [location_id, zone_type_1].  Required
        when ``cold_last=True``; ignored otherwise.
    cold_last
        When True, also verify AMBIENT → CHILLER → FREEZER ordering.
        Pass the same value that was given to ``route_all_pickruns()``.
    """
    report    = ValidationReport(n_routes=len(route_results))
    oneway_xs = graph.oneway_xs()

    item_weight: dict[str, float] = (
        dict(zip(items_df["item"], items_df["weight"])) if items_df is not None else {}
    )
    loc_zone: dict[str, str] = (
        dict(zip(locations_df["location_id"], locations_df["zone_type_1"]))
        if locations_df is not None and cold_last else {}
    )

    pr_groups = list(transactions.groupby("pickrun_no", sort=False))

    for idx, rr in enumerate(route_results):
        tiles  = rr.route_tiles
        pr_no  = pr_groups[idx][0] if idx < len(pr_groups) else f"route_{idx}"
        pr_txn = pr_groups[idx][1] if idx < len(pr_groups) else pd.DataFrame()

        # ── 1. Start and end at DEPOT ─────────────────────────────────────
        if not tiles or tiles[0] != DEPOT or tiles[-1] != DEPOT:
            start = repr(tiles[0])  if tiles         else "(empty)"
            end   = repr(tiles[-1]) if len(tiles) > 1 else "(empty)"
            report.depot_failures.append(
                f"pickrun {pr_no}: starts={start}  ends={end}"
            )

        # ── 2. Route completeness ─────────────────────────────────────────
        if not pr_txn.empty:
            expected  = {graph.loc_to_tile(lid) for lid in pr_txn["location_id"]}
            route_set = set(tiles)
            missing   = sorted(expected - route_set)
            if missing:
                report.completeness_failures.append(
                    f"pickrun {pr_no}: {len(missing)} tile(s) missing "
                    f"(e.g. {missing[:2]})"
                )

        # ── 3. Segment ordering ───────────────────────────────────────────
        # Strategy: for each (zone, weight_class) segment pair (A, B) where A
        # must come before B, the last route position of any A-tile must be
        # strictly less than the first route position of any B-tile.
        if item_weight and not pr_txn.empty:
            zones = COLD_ZONES if cold_last else ("AMBIENT",)

            # Bucket each location into its (zone, weight) segment
            seg_tiles: dict[tuple[str, str], set[str]] = {
                (z, w): set() for z in zones for w in _WEIGHT_CLS
            }
            for _, row in pr_txn.iterrows():
                tile  = graph.loc_to_tile(row["location_id"])
                w_cls = _weight_class(item_weight.get(row["item"], 0.0))
                z_cls = loc_zone.get(row["location_id"], "AMBIENT") if cold_last else "AMBIENT"
                if (z_cls, w_cls) in seg_tiles:
                    seg_tiles[(z_cls, w_cls)].add(tile)

            # Build a multi-index: tile → list of positions in the pick sequence
            pick_seq = [t for t in tiles if t != DEPOT]
            tile_positions: dict[str, list[int]] = defaultdict(list)
            for i, t in enumerate(pick_seq):
                tile_positions[t].append(i)

            # Compute span (first, last) of each segment's tile positions
            seg_span: dict[tuple[str, str], tuple[int, int]] = {}
            for key, tset in seg_tiles.items():
                positions = [p for t in tset for p in tile_positions[t]]
                if positions:
                    seg_span[key] = (min(positions), max(positions))

            # For each ordered pair of segments (A before B), check last_A < first_B
            seg_order = [(z, w) for z in zones for w in _WEIGHT_CLS]
            for i, key_a in enumerate(seg_order):
                if key_a not in seg_span:
                    continue
                for key_b in seg_order[i + 1:]:
                    if key_b not in seg_span:
                        continue
                    last_a, first_b = seg_span[key_a][1], seg_span[key_b][0]
                    if last_a >= first_b:
                        za, wa = key_a
                        zb, wb = key_b
                        report.ordering_violations.append(
                            f"pickrun {pr_no}: {za}·{wa} pick at position {last_a} "
                            f"must precede all {zb}·{wb} picks (first at position {first_b})"
                        )
                        break  # one violation message per pickrun is enough

        # ── 4. One-way aisle order (quality note) ─────────────────────────
        for i in range(len(tiles) - 1):
            xa, ya = _tile_xy(tiles[i])
            xb, yb = _tile_xy(tiles[i + 1])
            if (xa is not None and xa == xb
                    and xa in oneway_xs
                    and yb is not None and ya is not None
                    and yb < ya):
                report.oneway_suboptimal += 1

    return report
