"""End-to-end playground demo.

Generates synthetic warehouse data, compares two TSP variants, then sweeps
picker counts through the DES to show the makespan vs staffing curve.

Usage
-----
    python run_demo.py

Flags to try
------------
    # More pickruns for a longer simulation
    ds = generate(n_aisles=10, n_positions=20, n_items=300, n_pickruns=400, seed=42)

    # Compare TSP methods on a specific set of pickruns
    routes_nn   = route_all_pickruns(ds.transactions, graph, ds.items, improve=False)
    routes_2opt = route_all_pickruns(ds.transactions, graph, ds.items, improve=True)

    # Remove one-way constraints to see how much they cost
    stats = run_des(routes_nn, graph, n_pickers=3, model_aisle_contention=False)
"""
from __future__ import annotations

import time

from data_gen import generate
from graph import WarehouseGraph
from tsp import route_all_pickruns
from des import run_des


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_time(s: float) -> str:
    if s < 120:
        return f"{s:.1f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


def _pct(baseline: float, improved: float) -> str:
    if baseline == 0:
        return "  n/a"
    return f"{100.0 * (baseline - improved) / baseline:+.1f}%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 58)
    print("  Warehouse Optimisation Playground — Demo")
    print("=" * 58)

    # ------------------------------------------------------------------
    # 1. Synthetic data
    # ------------------------------------------------------------------
    print("\n[1/4] Generating synthetic data...")
    # n_days=1: all pickruns land on one day so the DES models a realistic
    # single-shift scenario (release times spread over 06:00–22:00).
    # 500 pickruns with avg 2.4 min each = ~20 h of work in a 16 h window,
    # so 1 picker runs behind and 2+ pickers just keep up — a clear knee.
    ds = generate(n_aisles=10, n_positions=20, n_items=300, n_pickruns=500, n_days=1, seed=42)
    n_pr    = ds.transactions["pickrun_no"].nunique()
    avg_ln  = len(ds.transactions) / max(1, n_pr)
    inv_pct = 100.0 * len(ds.inventory) / max(1, len(ds.items))
    print(f"      Items:        {len(ds.items)}  ({inv_pct:.0f}% assigned to locations)")
    print(f"      Locations:    {len(ds.locations)}  (10 aisles × 20 positions)")
    print(f"      Pickruns:     {n_pr}  (avg {avg_ln:.1f} items/run)")

    # ------------------------------------------------------------------
    # 2. Warehouse graph
    # ------------------------------------------------------------------
    print("\n[2/4] Building warehouse graph...")
    t0 = time.time()
    graph = WarehouseGraph(ds.locations)
    elapsed = time.time() - t0
    ow = sorted(graph.oneway_xs())
    print(f"      Done in {elapsed:.3f}s")
    print(f"      One-way aisles: x ∈ {ow}  (southbound only)")

    # ------------------------------------------------------------------
    # 3. TSP comparison: NN vs NN + 2-opt
    # ------------------------------------------------------------------
    print("\n[3/4] Routing pickruns — comparing TSP methods...")
    print(f"\n      {'Method':<30}  {'Avg dist/run':>12}  {'Routed in':>9}")
    print(f"      {'-'*30}  {'-'*12}  {'-'*9}")

    routes: dict[str, list] = {}
    for label, improve in [("Nearest-neighbor (NN)", False), ("NN + 2-opt", True)]:
        t0 = time.time()
        # cap at 200 for the comparison — TSP quality doesn't depend on sample size
        rs = route_all_pickruns(ds.transactions, graph, ds.items,
                                max_pickruns=200, improve=improve)
        elapsed = time.time() - t0
        avg_dist = sum(r.total_dist_m for r in rs) / max(1, len(rs))
        routes[label] = rs
        print(f"      {label:<30}  {avg_dist:>10.1f} m  {elapsed:>7.2f}s")

    nn_avg   = sum(r.total_dist_m for r in routes["Nearest-neighbor (NN)"]) / max(1, len(routes["Nearest-neighbor (NN)"]))
    opt_avg  = sum(r.total_dist_m for r in routes["NN + 2-opt"])             / max(1, len(routes["NN + 2-opt"]))
    print(f"\n      2-opt improvement: {_pct(nn_avg, opt_avg).strip()} shorter routes on average")

    # ------------------------------------------------------------------
    # 4. DES: sweep n_pickers (using NN routes)
    # ------------------------------------------------------------------
    print("\n[4/4] DES simulation — sweeping picker count (all 500 pickruns, NN routes)...")
    print(f"\n      {'Pickers':>7}  {'Makespan':>10}  {'Avg/run':>10}  {'Aisle wait':>11}")
    print(f"      {'-'*7}  {'-'*10}  {'-'*10}  {'-'*11}")

    # Route all 500 for the DES (the 200-pickrun subset was just for the comparison)
    routes_nn = route_all_pickruns(ds.transactions, graph, ds.items, improve=False)
    prev_makespan: float | None = None
    for n_p in [1, 2, 3, 4, 5]:
        stats = run_des(routes_nn, graph, n_pickers=n_p)
        ms = stats["makespan_s"]
        gain = ""
        if prev_makespan is not None:
            gain = f"  ({_pct(prev_makespan, ms).strip()})"
        print(
            f"      {n_p:>7}  "
            f"{_fmt_time(ms):>10}  "
            f"{_fmt_time(stats['avg_pickrun_time_s']):>10}  "
            f"{stats['avg_one_way_wait_s']:>8.1f} s"
            f"{gain}"
        )
        prev_makespan = ms

    if stats["worst_aisle_x"] is not None:
        print(
            f"\n      Worst one-way aisle: x={stats['worst_aisle_x']}  "
            f"total wait {_fmt_time(stats['worst_aisle_wait_s'])}"
        )

    print("\nDone.  Edit tsp.py and des.py to experiment further.")


if __name__ == "__main__":
    main()
