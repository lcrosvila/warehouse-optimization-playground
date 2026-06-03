"""Synthetic warehouse data generator.

Produces a ``Dataset`` whose DataFrames have the same column schema as the
real database loader (``../data_loader.py``), so experiments can be swapped
to real data without changing the rest of the playground code.

Data schema (mirrors the real DB tables)
-----------------------------------------
items        : item, weight, box_qty, picks_30, picks_60, picks_trend,
               zone_type_1, zone_type_2, zone_type_3
locations    : location_id, x_coord, y_coord, weight_limit,
               zone_type_1, module_id
loc_features : location_id, time_to_start_point
item_links   : item, community_id, community_priority
inventory    : item, location_id, qty
transactions : pickrun_no, order_no, item, location_id, qty, day, timestamp

``time_to_start_point`` is the actual shortest-path distance from the depot,
computed via ``graph.depot_distances()``.

Self-contained — the only intra-playground dependency is ``graph.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from graph import depot_distances

# Zone distribution mirroring a typical ambient/cold/frozen warehouse
ZONES = ["AMBIENT", "CHILLER", "FREEZER"]
ZONE_WEIGHTS = [0.70, 0.20, 0.10]


@dataclass
class Dataset:
    """Mirrors ``data.Dataset`` in the main repository."""
    items: pd.DataFrame
    locations: pd.DataFrame
    loc_features: pd.DataFrame
    item_links: pd.DataFrame
    inventory: pd.DataFrame
    transactions: pd.DataFrame


def generate(
    n_aisles: int = 10,
    n_positions: int = 20,
    n_items: int = 300,
    n_pickruns: int = 500,
    max_lines_per_run: int = 8,
    n_days: int = 90,
    seed: int = 42,
) -> Dataset:
    """Generate a synthetic warehouse dataset.

    Parameters
    ----------
    n_aisles
        Number of pick aisles (x dimension).  Odd-x aisles are one-way
        (northbound only) — the same convention as the production graph.
    n_positions
        Pick positions per aisle (y dimension).
    n_items
        Number of distinct SKUs.
    n_pickruns
        Number of pick orders to generate.
    max_lines_per_run
        Maximum picks per order (actual count drawn uniformly from [2, max]).
    n_days
        Days to spread pickruns over.
    seed
        Random seed for reproducibility.

    Notes
    -----
    The inventory assignment is zone-compatible and weight-respecting:
    heavier items are placed in higher-capacity locations within each zone.
    Items that cannot be placed (no zone-matching location with enough
    capacity) are silently dropped from inventory and transactions.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Items: Pareto demand distribution (80/20 fast/slow movers)
    # ------------------------------------------------------------------
    picks_60 = (rng.pareto(1.2, n_items) * 8).astype(int) + 1
    trend = rng.normal(1.0, 0.25, n_items).clip(0.3, 2.0)
    picks_30 = np.clip(picks_60 * 0.5 * trend + rng.normal(0, 1, n_items), 0, None).astype(int)

    items = pd.DataFrame({
        "item":        [f"ITEM_{i:04d}" for i in range(n_items)],
        "weight":      rng.uniform(0.5, 1000.0, n_items).round(1),
        "box_qty":     rng.integers(1, 25, n_items),
        "picks_30":    picks_30,
        "picks_60":    picks_60,
        "picks_trend": (picks_30 / np.maximum(picks_60, 1)).round(3),
        "zone_type_1": rng.choice(ZONES, n_items, p=ZONE_WEIGHTS),
        "zone_type_2": "STANDARD",
        "zone_type_3": "STANDARD",
    })

    # ------------------------------------------------------------------
    # Locations: rectangular aisle grid
    #
    # Zone assignment mirrors real warehouses: ambient aisles first, then
    # chiller, then freezer.  Module IDs group a few aisles together
    # (useful for explainability experiments).
    # ------------------------------------------------------------------
    n_ambient = max(1, int(n_aisles * 0.70))
    n_chiller = max(0, int(n_aisles * 0.20))

    loc_ids, xs, ys, zones_l, modules = [], [], [], [], []
    for aisle in range(1, n_aisles + 1):
        module = f"MOD_{chr(64 + (aisle - 1) % 26 + 1)}"
        if aisle <= n_ambient:
            zone = "AMBIENT"
        elif aisle <= n_ambient + n_chiller:
            zone = "CHILLER"
        else:
            zone = "FREEZER"
        for pos in range(1, n_positions + 1):
            loc_ids.append(f"A{aisle:02d}P{pos:03d}")
            xs.append(aisle)
            ys.append(pos)
            zones_l.append(zone)
            modules.append(module)

    n_locs = len(loc_ids)
    locations = pd.DataFrame({
        "location_id":  loc_ids,
        "x_coord":      xs,
        "y_coord":      ys,
        "weight_limit": rng.uniform(50, 2500, n_locs).round(1),
        "zone_type_1":  zones_l,
        "module_id":    modules,
    })

    # ------------------------------------------------------------------
    # Location features: actual shortest-path distance from the depot
    # ------------------------------------------------------------------
    dist_map = depot_distances(locations)
    loc_features = pd.DataFrame({
        "location_id":        locations["location_id"],
        "time_to_start_point": [max(1.0, dist_map[lid]) for lid in locations["location_id"]],
    })

    # ------------------------------------------------------------------
    # Item communities (groups of SKUs that tend to be co-picked)
    # ------------------------------------------------------------------
    n_comm = max(1, n_items // 30)
    sample_size = min(n_items, n_comm * 4)
    link_items = items["item"].sample(sample_size, random_state=int(rng.integers(0, 10**9)))
    item_links = pd.DataFrame({
        "item":               link_items.values,
        "community_id":       rng.integers(1, n_comm + 1, sample_size),
        "community_priority": rng.integers(1, 25, sample_size),
    })

    # ------------------------------------------------------------------
    # Inventory: zone-compatible, weight-respecting 1-to-1 assignment
    #
    # Within each zone, items are sorted heaviest-first and locations are
    # sorted by descending weight_limit — so the heaviest item gets the
    # highest-capacity slot.  Any item whose weight exceeds the available
    # slot capacity is skipped (no inventory entry, never picked).
    # ------------------------------------------------------------------
    inv_rows: list[dict] = []
    for zone in ZONES:
        z_items = items[items["zone_type_1"] == zone].sort_values("weight", ascending=False)
        z_locs  = locations[locations["zone_type_1"] == zone].sort_values("weight_limit", ascending=False)
        for (_, item_row), (_, loc_row) in zip(z_items.iterrows(), z_locs.iterrows()):
            if item_row["weight"] <= loc_row["weight_limit"]:
                qty = float(rng.integers(1, int(item_row["box_qty"]) + 1))
                inv_rows.append({
                    "item":        item_row["item"],
                    "location_id": loc_row["location_id"],
                    "qty":         qty,
                })

    inventory = pd.DataFrame(inv_rows)

    # ------------------------------------------------------------------
    # Transactions with timestamps
    #
    # Each pickrun has a release time drawn uniformly within a 16-hour
    # working window (06:00–22:00).  Timestamps let the DES model real
    # release staggering rather than releasing all orders at t=0.
    # ------------------------------------------------------------------
    item_to_loc  = dict(zip(inventory["item"],     inventory["location_id"]))
    box_qty_map  = dict(zip(items["item"],         items["box_qty"]))
    base_date    = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
    day_span_s   = 16 * 3600  # 16-hour picking window

    pickrun_days    = np.sort(rng.integers(0, n_days, n_pickruns))
    n_lines_arr     = rng.integers(2, max_lines_per_run + 1, n_pickruns)
    release_offsets = rng.uniform(0, day_span_s, n_pickruns)
    item_arr        = inventory["item"].values  # only assigned items can be picked

    tx_rows: list[dict] = []
    for pr_idx, (n_ln, day, offset) in enumerate(
        zip(n_lines_arr, pickrun_days, release_offsets), start=1
    ):
        pick_items = rng.choice(item_arr, size=int(n_ln), replace=False)
        ts = base_date + timedelta(days=int(day), seconds=float(offset))
        pr_no = f"PR_{pr_idx:05d}"
        for itm in pick_items:
            tx_rows.append({
                "pickrun_no":  pr_no,
                "order_no":    pr_no,   # same as pickrun_no in synthetic data
                "item":        itm,
                "location_id": item_to_loc[itm],
                "qty":         int(rng.integers(1, int(box_qty_map[itm]) + 1)),
                "day":         int(day),
                "timestamp":   ts,
            })

    transactions = (
        pd.DataFrame(tx_rows)
        .sort_values(["day", "pickrun_no"])
        .reset_index(drop=True)
    )

    return Dataset(
        items=items,
        locations=locations,
        loc_features=loc_features,
        item_links=item_links,
        inventory=inventory,
        transactions=transactions,
    )
