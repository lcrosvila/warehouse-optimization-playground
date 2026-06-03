"""Discrete-event simulation of warehouse picking operations.

Model
-----
  • A pool of ``n_pickers`` pickers, each taking one pickrun at a time.
  • One-way aisles are modelled as SimPy Resources with capacity=1:
    only one picker may traverse a one-way aisle at a time.  Others
    queue at the aisle entrance until it is free.
  • Pickruns are released into the system according to their timestamps
    (``release_s`` in RouteResult).  All pickruns at t=0 if no timestamps.

Extensions (all optional, disabled by default)
----------------------------------------------
  n_replenishers      — worker pool that restocks empty slots; pickers wait
                        until a replenisher services the slot before picking.
                        Tune ``replenish_prob`` (fraction of picks that hit an
                        empty slot) and ``restock_time_s`` (seconds to restock).
  fatigue_pct_per_100 — picker walking speed degrades linearly; after every 100
                        picks the picker walks this many % slower.  Models shift-
                        end fatigue or equipment battery drain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import simpy

from graph import WarehouseGraph, DEPOT
from tsp import RouteResult, SPEED_M_S, PICK_TIME_S


# ---------------------------------------------------------------------------
# Machine profiles
# ---------------------------------------------------------------------------

@dataclass
class MachineProfile:
    """Physical characteristics of a picking machine.

    Attributes
    ----------
    name
        Display name for reporting.
    speed_m_s
        Travel speed in metres/second.
    pick_time_s
        Seconds to pick one item from its slot.
    narrow_ok
        True if the machine can enter one-way (narrow) aisles and join the
        aisle queue.  False for wide machines (e.g. counterbalance forklifts)
        that must detour via the cross-aisles; they pay ``detour_factor``
        times the normal aisle travel time instead.
    detour_factor
        Only used when ``narrow_ok=False``.  Multiplier applied to the
        distance of any narrow-aisle segment to model the longer route around.
    """
    name:          str
    speed_m_s:     float
    pick_time_s:   float
    narrow_ok:     bool  = True
    detour_factor: float = 2.0


# Pre-built profiles — use directly or construct your own.
HUMAN          = MachineProfile("human",          speed_m_s=1.5,  pick_time_s=4.0)
REACH_TRUCK    = MachineProfile("reach_truck",    speed_m_s=2.5,  pick_time_s=7.0)
COUNTERBALANCE = MachineProfile("counterbalance", speed_m_s=3.5,  pick_time_s=10.0,
                                narrow_ok=False, detour_factor=2.2)


def run_des(
    route_results: list[RouteResult],
    graph: WarehouseGraph,
    n_pickers: int = 1,
    model_aisle_contention: bool = True,
    n_replenishers: int = 0,
    replenish_prob: float = 0.0,
    restock_time_s: float = 30.0,
    fatigue_pct_per_100_picks: float = 0.0,
    machine_profile: MachineProfile | None = None,
    reroute_on_wait_s: float = 0.0,
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
    n_replenishers
        Number of dedicated replenishment workers.  When > 0 and
        ``replenish_prob`` > 0, pickers occasionally find an empty slot and
        must wait for a replenisher to restock it.
    replenish_prob
        Probability (0–1) that any given pick location has an empty slot
        requiring a restock before the item can be picked.
    restock_time_s
        Time in seconds for a replenisher to service one empty slot.
    fatigue_pct_per_100_picks
        Each picker's walking speed decreases by this many percent for every
        100 picks completed.  E.g. 5.0 means after 200 picks the picker walks
        10% slower.  Speed is floored at 10% of the baseline.

    machine_profile : MachineProfile | None
        Overrides the per-picker speed and pick time.  Use the preset constants
        ``HUMAN``, ``REACH_TRUCK``, or ``COUNTERBALANCE``, or build your own
        with ``MachineProfile(...)``.  ``None`` falls back to the module-level
        ``SPEED_M_S`` / ``PICK_TIME_S`` constants (equivalent to ``HUMAN``).
    reroute_on_wait_s : float
        When > 0 and a picker has been queued for a one-way aisle for longer
        than this many seconds, they give up waiting and defer all tiles in
        that aisle to the end of their route, processing other aisles first.
        Set to e.g. 30.0 to explore whether opportunistic rerouting reduces
        makespan.  Experiment: does rerouting help more with 2 pickers or 5?

    Returns
    -------
    dict with keys:
        makespan_s              — wall-clock time until the last pickrun finishes
        avg_pickrun_time_s      — mean time a picker spends on one pickrun
        avg_one_way_wait_s      — mean wait time per aisle-entry event
        worst_aisle_x           — x-column with the longest total wait (or None)
        worst_aisle_wait_s      — total wait accumulated in worst_aisle_x
        total_restock_wait_s    — total seconds all pickers spent waiting for restocks
        avg_restock_wait_s      — mean restock-wait per individual restock event
        n_restock_events        — number of restock waits that occurred
        n_reroutes              — number of times a picker was rerouted mid-run
    """
    mp        = machine_profile or HUMAN
    env       = simpy.Environment()
    oneway_xs = graph.oneway_xs()
    rng       = np.random.default_rng(0)

    picker_pool = simpy.Resource(env, capacity=max(1, n_pickers))
    aisle_res: dict[int, simpy.Resource] = (
        {x: simpy.Resource(env, capacity=1) for x in oneway_xs}
        if model_aisle_contention
        else {}
    )

    # Replenishment: a Store holds pending restock events; replenisher workers
    # pull one event, sleep for restock_time_s, then trigger the event so the
    # waiting picker can proceed.
    restock_queue: simpy.Store = simpy.Store(env)

    aisle_wait:    dict[int, list[float]] = {x: [] for x in oneway_xs}
    pickrun_times: list[float] = []
    restock_waits: list[float] = []
    reroute_count: list[int]   = [0]   # mutable counter shared across closures

    # --- Replenisher worker processes ---
    def _replenisher_worker():
        while True:
            done_event = yield restock_queue.get()
            yield env.timeout(restock_time_s)
            done_event.succeed()

    if n_replenishers > 0 and replenish_prob > 0.0:
        for _ in range(n_replenishers):
            env.process(_replenisher_worker())

    # --- Picker processes ---
    def picker_process(rr: RouteResult):
        yield env.timeout(rr.release_s)
        with picker_pool.request() as req:
            yield req
            yield from _do_route(rr)

    def _do_route(rr: RouteResult):
        t_start    = env.now
        tiles      = list(rr.route_tiles)   # mutable copy so rerouting can reorder
        prev       = tiles[0]
        picks_done = 0
        i          = 1

        while i < len(tiles):
            tile = tiles[i]
            x    = graph.tile_x(tile)

            # Speed: machine profile base, then linearly degraded by fatigue
            eff_speed = mp.speed_m_s
            if fatigue_pct_per_100_picks > 0.0:
                reduction = (picks_done / 100.0) * (fatigue_pct_per_100_picks / 100.0)
                eff_speed = mp.speed_m_s * max(0.1, 1.0 - reduction)

            if x is not None and x in aisle_res:
                # Collect the full contiguous segment in this one-way aisle
                seg: list[str] = []
                j = i
                while j < len(tiles) and graph.tile_x(tiles[j]) == x:
                    seg.append(tiles[j])
                    j += 1

                if not mp.narrow_ok:
                    # Wide machine: can't enter — pay a detour penalty instead
                    # of queuing for the resource.
                    for t in seg:
                        d = graph.distance(prev, t) * mp.detour_factor
                        yield env.timeout(d / eff_speed)
                        if t != graph.end_tile:
                            yield from _pick_at(t)
                            picks_done += 1
                        prev = t
                    i = j
                elif reroute_on_wait_s > 0 and len(aisle_res[x].queue) > 0:
                    # Rerouting enabled and aisle is queued — defer this aisle's
                    # tiles to the end of the route and continue with other aisles.
                    deferred = [t for t in tiles[i:] if graph.tile_x(t) == x]
                    rest     = [t for t in tiles[i:] if graph.tile_x(t) != x]
                    tiles[i:] = rest + deferred
                    reroute_count[0] += 1
                    # Do NOT advance i — re-evaluate tiles[i] (now a different tile)
                else:
                    t_req = env.now
                    with aisle_res[x].request() as req:
                        yield req
                        aisle_wait[x].append(env.now - t_req)
                        for t in seg:
                            yield env.timeout(graph.distance(prev, t) / eff_speed)
                            if t != graph.end_tile:
                                yield from _pick_at(t)
                                picks_done += 1
                            prev = t
                    i = j
            else:
                yield env.timeout(graph.distance(prev, tile) / eff_speed)
                if tile != graph.end_tile:
                    yield from _pick_at(tile)
                    picks_done += 1
                prev = tile
                i   += 1

        pickrun_times.append(env.now - t_start)

    def _pick_at(tile: str):
        """Yield pick-time events, inserting a restock wait when the slot is empty."""
        if (n_replenishers > 0
                and replenish_prob > 0.0
                and rng.random() < replenish_prob):
            t_req      = env.now
            done_event = env.event()
            yield restock_queue.put(done_event)
            yield done_event
            restock_waits.append(env.now - t_req)
        yield env.timeout(PICK_TIME_S)

    for rr in route_results:
        env.process(picker_process(rr))

    env.run()

    makespan  = float(env.now)
    avg_time  = float(np.mean(pickrun_times)) if pickrun_times else 0.0
    all_waits = [w for ws in aisle_wait.values() for w in ws]
    avg_wait  = float(np.mean(all_waits)) if all_waits else 0.0

    worst_x:    int | None = None
    worst_wait: float      = 0.0
    if aisle_wait:
        totals = {x: sum(ws) for x, ws in aisle_wait.items() if ws}
        if totals:
            worst_x    = max(totals, key=totals.get)
            worst_wait = totals[worst_x]

    total_rw = sum(restock_waits)
    avg_rw   = float(np.mean(restock_waits)) if restock_waits else 0.0

    return {
        "makespan_s":           makespan,
        "avg_pickrun_time_s":   avg_time,
        "avg_one_way_wait_s":   avg_wait,
        "worst_aisle_x":        worst_x,
        "worst_aisle_wait_s":   worst_wait,
        "total_restock_wait_s": total_rw,
        "avg_restock_wait_s":   avg_rw,
        "n_restock_events":     len(restock_waits),
        "n_reroutes":           reroute_count[0],
    }
