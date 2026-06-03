"""Discrete-event simulation of warehouse picking operations.

Model
-----
  • A pool of ``n_pickers`` pickers, each taking one pickrun at a time.
  • One-way aisles are modelled as SimPy Resources with capacity=1:
    only one picker may traverse a one-way aisle at a time.  Others
    queue at the aisle entrance until it is free.
  • Pickruns are released into the system according to their timestamps
    (``release_s`` in RouteResult).  All pickruns at t=0 if no timestamps.

The DES re-computes travel time tile-by-tile from the route sequence and
the graph's distance function, so it correctly reflects the actual path
chosen by the TSP solver.

Your task
---------
  • Sweep ``n_pickers`` and plot makespan to find the throughput knee.
  • Adjust ``model_aisle_contention=False`` to remove one-way constraints
    and quantify the contention overhead.
  • Extend the model: add a replenishment queue, picker fatigue (slower
    speed after N picks), or a zone-crossing penalty.
"""
from __future__ import annotations

import numpy as np
import simpy

from graph import WarehouseGraph, DEPOT
from tsp import RouteResult, SPEED_M_S, PICK_TIME_S


def run_des(
    route_results: list[RouteResult],
    graph: WarehouseGraph,
    n_pickers: int = 1,
    model_aisle_contention: bool = True,
) -> dict:
    """Simulate all pickruns and return aggregate statistics.

    Parameters
    ----------
    route_results
        List of routed pickruns from ``tsp.route_all_pickruns()``.
    graph
        WarehouseGraph — needed for tile classification and distances.
    n_pickers
        Number of concurrent pickers sharing the warehouse.
    model_aisle_contention
        When True, odd-x (one-way) aisles have capacity=1: only one picker
        at a time, others queue.  Set False to measure the ideal lower bound
        with no aisle contention.

    Returns
    -------
    dict with keys:
        makespan_s          — wall-clock time until the last pickrun finishes
        avg_pickrun_time_s  — mean time a picker spends on one pickrun
        avg_one_way_wait_s  — mean wait time per aisle-entry event
        worst_aisle_x       — x-column with the longest total wait (or None)
        worst_aisle_wait_s  — total wait accumulated in worst_aisle_x
    """
    env = simpy.Environment()
    oneway_xs = graph.oneway_xs()

    picker_pool = simpy.Resource(env, capacity=max(1, n_pickers))
    aisle_res: dict[int, simpy.Resource] = (
        {x: simpy.Resource(env, capacity=1) for x in oneway_xs}
        if model_aisle_contention
        else {}
    )

    aisle_wait: dict[int, list[float]] = {x: [] for x in oneway_xs}
    pickrun_times: list[float] = []

    def picker_process(rr: RouteResult):
        # Wait until this pickrun is released into the system
        yield env.timeout(rr.release_s)
        with picker_pool.request() as req:
            yield req
            yield from _do_route(rr)

    def _do_route(rr: RouteResult):
        t_start = env.now
        tiles = rr.route_tiles
        prev = tiles[0]   # start tile (DEPOT) — travel origin, not a pick stop
        i = 1

        while i < len(tiles):
            tile = tiles[i]
            x = graph.tile_x(tile)

            if x is not None and x in aisle_res:
                # Collect the full contiguous segment in this one-way aisle
                seg: list[str] = []
                while i < len(tiles) and graph.tile_x(tiles[i]) == x:
                    seg.append(tiles[i])
                    i += 1

                # Acquire the aisle resource for the entire segment — simulates
                # a picker claiming the aisle until they exit at the far end.
                t_req = env.now
                with aisle_res[x].request() as req:
                    yield req
                    aisle_wait[x].append(env.now - t_req)
                    for t in seg:
                        yield env.timeout(graph.distance(prev, t) / SPEED_M_S)
                        if t != graph.end_tile:
                            yield env.timeout(PICK_TIME_S)
                        prev = t
            else:
                yield env.timeout(graph.distance(prev, tile) / SPEED_M_S)
                if tile != graph.end_tile:
                    yield env.timeout(PICK_TIME_S)
                prev = tile
                i += 1

        pickrun_times.append(env.now - t_start)

    for rr in route_results:
        env.process(picker_process(rr))

    env.run()

    makespan   = float(env.now)
    avg_time   = float(np.mean(pickrun_times)) if pickrun_times else 0.0
    all_waits  = [w for ws in aisle_wait.values() for w in ws]
    avg_wait   = float(np.mean(all_waits)) if all_waits else 0.0

    worst_x:    int | None = None
    worst_wait: float      = 0.0
    if aisle_wait:
        totals = {x: sum(ws) for x, ws in aisle_wait.items() if ws}
        if totals:
            worst_x    = max(totals, key=totals.get)
            worst_wait = totals[worst_x]

    return {
        "makespan_s":          makespan,
        "avg_pickrun_time_s":  avg_time,
        "avg_one_way_wait_s":  avg_wait,
        "worst_aisle_x":       worst_x,
        "worst_aisle_wait_s":  worst_wait,
    }
