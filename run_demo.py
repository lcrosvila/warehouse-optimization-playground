"""End-to-end playground demo.

Generates synthetic warehouse data, compares TSP solvers, then runs DES
scenarios that show picker-count scaling, replenishment contention, and
picker fatigue.

Usage
-----
    python run_demo.py

Things to try
-------------
    # Swap solver in the TSP comparison
    rs = route_all_pickruns(ds.transactions, graph, ds.items,
                            max_pickruns=200, solver="sa")

    # Remove one-way aisle constraints
    stats = run_des(routes, graph, n_pickers=3, model_aisle_contention=False)

    # Add a replenishment team and 10% empty-slot probability
    stats = run_des(routes, graph, n_pickers=3,
                    n_replenishers=2, replenish_prob=0.10, restock_time_s=25)

    # Model picker fatigue (5% slower per 100 picks)
    stats = run_des(routes, graph, n_pickers=3, fatigue_pct_per_100_picks=5.0)
"""
from __future__ import annotations

import time

from data_gen import generate
from graph import WarehouseGraph
from tsp import route_all_pickruns
from des import run_des, HUMAN, REACH_TRUCK, COUNTERBALANCE
from sanity import validate_routes


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
    print("=" * 62)
    print("  Warehouse Optimisation Playground — Demo")
    print("=" * 62)

    # ------------------------------------------------------------------
    # 1. Synthetic data
    # ------------------------------------------------------------------
    print("\n[1/5] Generating synthetic data...")
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
    print("\n[2/5] Building warehouse graph...")
    t0 = time.time()
    graph = WarehouseGraph(ds.locations)
    elapsed = time.time() - t0
    ow = sorted(graph.oneway_xs())
    print(f"      Done in {elapsed:.3f}s")
    print(f"      One-way aisles: x ∈ {ow}")

    # ------------------------------------------------------------------
    # 3. TSP solver comparison (200-pickrun sample)
    # ------------------------------------------------------------------
    print("\n[3/6] TSP solver comparison (200 pickruns each)...")
    print(f"\n      {'Solver':<22}  {'Avg dist/run':>12}  {'vs NN':>7}  {'Time':>7}  {'Order fixes':>12}")
    print(f"      {'-'*22}  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*12}")

    SOLVER_LABELS = [
        ("nn",        "Nearest-neighbor",          dict()),
        ("2opt",      "NN + 2-opt",                dict()),
        ("or_opt",    "NN + or-opt",               dict()),
        ("aisle_nn",  "Aisle-sorted NN",           dict()),
        ("bucketed",  "Bucketed brute",            dict()),
        ("sa",        "Simul. annealing",          dict()),
        ("mst",       "MST Christofides",          dict()),
        ("aco",       "Ant Colony (ACO)",          dict()),
        # Same solver but with cold-chain ordering enforced
        ("nn",        "NN (cold-last)",            dict(cold_last=True)),
        ("2opt",      "NN + 2-opt (cold-last)",    dict(cold_last=True)),
    ]

    routes_per_solver: dict[str, list] = {}
    nn_avg: float = 0.0
    for key, label, extra_kw in SOLVER_LABELS:
        t0 = time.time()
        try:
            rs = route_all_pickruns(
                ds.transactions, graph, ds.items,
                locations_df=ds.locations,
                max_pickruns=200, solver=key,
                **extra_kw,
            )
            elapsed  = time.time() - t0
            avg_dist = sum(r.total_dist_m for r in rs) / max(1, len(rs))
            avg_fixes = sum(r.ordering_fixes for r in rs) / max(1, len(rs))
            cache_key = label
            routes_per_solver[cache_key] = rs
            vs = f"{_pct(nn_avg, avg_dist).strip():>7}" if nn_avg else "  base"
            print(f"      {label:<22}  {avg_dist:>10.1f} m  {vs}  {elapsed:>5.2f}s  {avg_fixes:>10.1f}")
            if key == "nn" and not extra_kw:
                nn_avg = avg_dist
        except ImportError as exc:
            print(f"      {label:<22}  skipped ({exc})")

    # ------------------------------------------------------------------
    # 4. DES picker-count sweep (NN routes, all 500 pickruns)
    # ------------------------------------------------------------------
    print("\n[4/6] DES — sweeping picker count (NN routes, 500 pickruns)...")
    print(f"\n      {'Pickers':>7}  {'Makespan':>10}  {'Avg/run':>10}  {'Aisle wait':>11}")
    print(f"      {'-'*7}  {'-'*10}  {'-'*10}  {'-'*11}")

    routes_nn   = route_all_pickruns(ds.transactions, graph, ds.items,
                                     locations_df=ds.locations, solver="nn")
    prev_ms: float | None = None
    last_stats: dict = {}
    for n_p in [1, 2, 3, 4, 5]:
        stats    = run_des(routes_nn, graph, n_pickers=n_p)
        ms       = stats["makespan_s"]
        gain     = f"  ({_pct(prev_ms, ms).strip()})" if prev_ms else ""
        print(
            f"      {n_p:>7}  "
            f"{_fmt_time(ms):>10}  "
            f"{_fmt_time(stats['avg_pickrun_time_s']):>10}  "
            f"{stats['avg_one_way_wait_s']:>8.1f} s"
            f"{gain}"
        )
        prev_ms    = ms
        last_stats = stats

    if last_stats.get("worst_aisle_x") is not None:
        print(
            f"\n      Worst one-way aisle: x={last_stats['worst_aisle_x']}  "
            f"total wait {_fmt_time(last_stats['worst_aisle_wait_s'])}"
        )

    # ------------------------------------------------------------------
    # 5. DES extension scenarios (3 pickers, NN routes)
    # ------------------------------------------------------------------
    print("\n[5/6] DES extensions at 3 pickers...")
    print(f"\n      {'Scenario':<32}  {'Makespan':>10}  {'Detail':>20}")
    print(f"      {'-'*32}  {'-'*10}  {'-'*20}")

    scenarios = [
        ("Baseline",                  dict()),
        ("No aisle contention",       dict(model_aisle_contention=False)),
        ("Replenish 2w / 5% empty",   dict(n_replenishers=2, replenish_prob=0.05)),
        ("Replenish 2w / 15% empty",  dict(n_replenishers=2, replenish_prob=0.15)),
        ("Fatigue 5% per 100 picks",  dict(fatigue_pct_per_100_picks=5.0)),
        ("Fatigue 10% per 100 picks", dict(fatigue_pct_per_100_picks=10.0)),
        ("Reroute on 30s wait",       dict(reroute_on_wait_s=30.0)),
        ("Reroute on 10s wait",       dict(reroute_on_wait_s=10.0)),
    ]

    for label, kwargs in scenarios:
        s   = run_des(routes_nn, graph, n_pickers=3, **kwargs)
        rw  = s["n_restock_events"]
        nr  = s["n_reroutes"]
        if rw:
            detail = f"{s['total_restock_wait_s'] / 60:.1f} min ({rw} restocks)"
        elif nr:
            detail = f"{nr} reroutes"
        else:
            detail = "—"
        print(f"      {label:<32}  {_fmt_time(s['makespan_s']):>10}  {detail:>20}")

    # ------------------------------------------------------------------
    # 6. Machinery comparison (3 pickers, NN routes)
    # ------------------------------------------------------------------
    print("\n[6/6] Machinery comparison at 3 pickers...")
    print(f"\n      {'Machine':<18}  {'Makespan':>10}  {'Avg/run':>10}  {'Aisle wait':>11}")
    print(f"      {'-'*18}  {'-'*10}  {'-'*10}  {'-'*11}")

    for profile in [HUMAN, REACH_TRUCK, COUNTERBALANCE]:
        s = run_des(routes_nn, graph, n_pickers=3, machine_profile=profile)
        print(
            f"      {profile.name:<18}  "
            f"{_fmt_time(s['makespan_s']):>10}  "
            f"{_fmt_time(s['avg_pickrun_time_s']):>10}  "
            f"{s['avg_one_way_wait_s']:>8.1f} s"
        )

    # ------------------------------------------------------------------
    # 7. Sanity checks
    # ------------------------------------------------------------------
    print("\n[7/7] Sanity checks...")

    # Scenario A: standard NN routes — weight ordering only
    print("\n  A) NN routes, weight ordering (cold_last=False):")
    report_a = validate_routes(routes_nn, ds.transactions, graph,
                                ds.items, ds.locations, cold_last=False)
    print(report_a.summary())

    # Scenario B: same NN routes, but now also validate cold-chain ordering.
    # These routes were NOT generated with cold_last=True, so we expect
    # cold-chain violations to be caught here.
    print("  B) NN routes checked against cold-chain ordering (cold_last=False routes):")
    report_b = validate_routes(routes_nn, ds.transactions, graph,
                                ds.items, ds.locations, cold_last=True)
    print(report_b.summary())
    if not report_b.passed:
        print("  ^ Expected FAIL — routes were generated without cold_last=True.")

    # Scenario C: re-route with cold_last=True and confirm all constraints pass
    print("  C) Cold-last NN routes, full constraint check (cold_last=True):")
    routes_cold = route_all_pickruns(ds.transactions, graph, ds.items,
                                     locations_df=ds.locations,
                                     solver="nn", cold_last=True)
    report_c = validate_routes(routes_cold, ds.transactions, graph,
                                ds.items, ds.locations, cold_last=True)
    print(report_c.summary())

    print("\nDone.  Edit tsp.py, des.py, and sanity.py to experiment further.")


if __name__ == "__main__":
    main()
