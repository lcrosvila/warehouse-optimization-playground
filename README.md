# Warehouse Optimisation Playground

A self-contained Python sandbox for experimenting with **TSP** (Travelling
Salesman Problem) routing and **DES** (Discrete-Event Simulation) for
warehouse pick-route optimisation.

## Quick start

```bash
pip install -r requirements.txt
/path/to/python run_demo.py
```

Expected output: a solver comparison table, a picker-count sweep, DES
extension scenarios, and a machinery comparison.

---

## Background

In a warehouse, a *pickrun* is one customer order: a picker walks a route and
collects a set of items from their shelf locations.  Two problems drive most
of the optimisation work:

1. **TSP** — in what order should the picker visit the locations to minimise
   total walking distance?
2. **DES** — given several concurrent pickers and shared resources (aisles,
   replenishment workers), what is the real-world *makespan* and where are
   the bottlenecks?

The warehouse is a rectangular grid.  *One-way aisles* (odd-x columns, south →
north only) prevent head-on collisions but create queuing when multiple pickers
want the same aisle simultaneously.

Pick-ordering constraints apply for ergonomic and damage-prevention reasons:

| Segment | Class    | Rule |
|---------|----------|------|
| 1st     | Heavy    | weight ≥ 500 kg — picked first (sits at the bottom of the cart) |
| 2nd     | Normal   | everything else |
| 3rd     | Fragile  | weight ≤ 50 kg — picked last (rests on top) |

Each segment is routed independently, so the ordering guarantee is hard.

---

## Files

| File | Role |
|------|------|
| `graph.py` | Directed routing graph: one-way aisles, depot, Dijkstra distances |
| `data_gen.py` | Synthetic data generator |
| `tsp.py` | Eight TSP solvers with a unified `solver=` interface |
| `des.py` | SimPy DES: picker pool, aisle contention, replenishment, fatigue, machinery |
| `sanity.py` | Independent constraint checker — verifies routes without modifying them |
| `run_demo.py` | End-to-end demo — run this to see everything working |


---

## TSP solvers

Pass `solver=` to `route_all_pickruns()` (or `route_pickrun()`).

| Key | Algorithm | Notes |
|-----|-----------|-------|
| `"nn"` | Nearest-neighbor | Greedy O(n²) baseline (default) |
| `"2opt"` | NN + 2-opt | Reverses sub-sequences; ~5–15% shorter routes |
| `"or_opt"` | NN + or-opt | Relocates chains of 1–2 tiles to better gaps; often beats 2-opt |
| `"aisle_nn"` | S-shape sweep | Sorts by aisle column, alternates direction; fast, layout-aware |
| `"bucketed"` | Bucketed brute | NN picks buckets of 6; brute-forces permutations within each |
| `"sa"` | Simulated annealing | Random 2-opt moves with temperature cooling; escapes local optima |
| `"mst"` | MST Christofides | Minimum spanning tree + greedy odd-degree matching (needs scipy) |
| `"aco"` | Ant Colony (ACO) | Pheromone-guided probabilistic search; see exploration tasks below |

```python
from tsp import route_all_pickruns

# Swap one line to try a different solver
routes = route_all_pickruns(ds.transactions, graph, ds.items, solver="or_opt")

# Solvers accept keyword arguments forwarded from route_all_pickruns
routes = route_all_pickruns(..., solver="bucketed", bucket_size=5)
routes = route_all_pickruns(..., solver="sa", n_iter=8000, T_start=200.0)
routes = route_all_pickruns(..., solver="aco", n_ants=30, n_iterations=80)
```

### Cold-chain ordering

Pass `cold_last=True` (and `locations_df`) to enforce an additional ordering
constraint: AMBIENT items are always picked before CHILLER, and CHILLER before
FREEZER.  This prevents cold items from sitting in the picker's cart while the
picker walks warm aisles.

```python
routes = route_all_pickruns(
    ds.transactions, graph, ds.items,
    locations_df=ds.locations,   # needed to look up each location's zone
    solver="2opt",
    cold_last=True,
)
```

Combined with the weight constraint, this produces up to **9 ordered segments**
per pickrun: `(AMBIENT·heavy, AMBIENT·normal, AMBIENT·fragile, CHILLER·heavy,
…, FREEZER·fragile)`.  Each segment is routed independently, so both guarantees
are hard — no cold item will ever appear before all ambient items in the route.

The `ordering_fixes` field in `RouteResult` counts how many items in the
original transaction were out of order relative to the enforced constraints.
With `cold_last=True`, this number includes zone violations as well as weight
violations, so it will be higher than without — that's expected; it's telling
you how often the raw order data violates the cold-chain rule.

Note: because cold zones are physically grouped on one side of the warehouse,
enforcing cold-last sometimes *reduces* total route distance by clustering the
route spatially.  Compare the "NN (cold-last)" row in the solver table when you
run `run_demo.py`.

---

## DES parameters

`run_des()` accepts a growing set of optional parameters:

| Parameter | Default | What it models |
|-----------|---------|----------------|
| `n_pickers` | 1 | Concurrent pickers sharing the warehouse |
| `model_aisle_contention` | True | One-way aisles have capacity=1; others queue |
| `n_replenishers` | 0 | Dedicated workers who restock empty slots |
| `replenish_prob` | 0.0 | Fraction of picks that hit an empty slot |
| `restock_time_s` | 30.0 | Seconds to service one empty slot |
| `fatigue_pct_per_100_picks` | 0.0 | Speed reduction per 100 picks (linear) |
| `machine_profile` | `HUMAN` | Speed and pick-time profile for the picker fleet |
| `reroute_on_wait_s` | 0.0 | Re-order deferred tiles when aisle queue wait exceeds this |

### Machine profiles

Three presets are available in `des.py`; construct a custom one with
`MachineProfile(name, speed_m_s, pick_time_s, narrow_ok, detour_factor)`.

| Preset | Speed | Pick time | Narrow aisles |
|--------|-------|-----------|---------------|
| `HUMAN` | 1.5 m/s | 4 s | ✓ enters and queues |
| `REACH_TRUCK` | 2.5 m/s | 7 s | ✓ enters and queues |
| `COUNTERBALANCE` | 3.5 m/s | 10 s | ✗ pays 2.2× detour penalty |

```python
from des import run_des, REACH_TRUCK, MachineProfile

# Use a preset
stats = run_des(routes, graph, n_pickers=3, machine_profile=REACH_TRUCK)

# Build a custom profile (e.g. a slow electric pallet jack)
pallet_jack = MachineProfile("pallet_jack", speed_m_s=1.0, pick_time_s=6.0)
stats = run_des(routes, graph, n_pickers=4, machine_profile=pallet_jack)
```

---

## Data schema

`data_gen.generate()` returns a `Dataset` with six DataFrames.  The column
names match the real database loader exactly — swapping in real data requires
only replacing the `generate()` call.

| DataFrame | Key columns |
|-----------|-------------|
| `items` | `item`, `weight`, `box_qty`, `picks_30`, `picks_60`, `picks_trend`, `zone_type_1` |
| `locations` | `location_id`, `x_coord`, `y_coord`, `weight_limit`, `zone_type_1`, `module_id` |
| `loc_features` | `location_id`, `time_to_start_point` |
| `inventory` | `item`, `location_id`, `qty` |
| `transactions` | `pickrun_no`, `order_no`, `item`, `location_id`, `qty`, `day`, `timestamp` |
| `item_links` | `item`, `community_id`, `community_priority` |

---

## Working with the data

### What the generator already models realistically

`data_gen.generate()` is not just random noise — several properties were
deliberately tuned to match the production warehouse:

- **Pareto demand** (`picks_30`, `picks_60`): item velocity follows a Pareto
  distribution with shape 1.2, giving a roughly 80/20 fast/slow split.
  The `picks_trend` column (ratio of last-30-day to last-60-day picks) models
  seasonal acceleration or deceleration.
- **Zone layout** (70 % AMBIENT / 20 % CHILLER / 10 % FREEZER): aisles are
  grouped by temperature zone left-to-right, matching the physical layout of
  most cold-chain warehouses.
- **Weight-constrained slotting**: the heaviest items are assigned to the
  highest-capacity locations within each zone.  Items that exceed all available
  slot capacities are silently dropped — the same behaviour as the real slotting
  engine.
- **Staggered release times**: each pickrun gets a timestamp drawn uniformly
  within a 16-hour window (06:00–22:00), so the DES models realistic order
  waves rather than a single burst at t=0.

### Generator parameters

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `n_aisles` | 10 | Width of the warehouse (x dimension) |
| `n_positions` | 20 | Depth of each aisle (y dimension) |
| `n_items` | 300 | SKU count |
| `n_pickruns` | 500 | Orders to generate |
| `max_lines_per_run` | 8 | Upper bound on picks per order (uniform draw from [2, max]) |
| `n_days` | 90 | Spread pickruns across this many calendar days |
| `seed` | 42 | Random seed; change it to get a different-but-equally-valid instance |

### Scenario recipes

**Busy single shift** — all 500 orders released on one day, creating realistic
queuing pressure in the DES:
```python
ds = generate(n_pickruns=500, n_days=1, seed=42)
```

**Large warehouse** — 20 aisles, 40 positions, more SKUs:
```python
ds = generate(n_aisles=20, n_positions=40, n_items=800, n_pickruns=1000)
```

**Small, dense warehouse** — few aisles, long pickruns, high contention:
```python
ds = generate(n_aisles=4, n_positions=30, n_items=100, max_lines_per_run=15)
```

**Heavy-goods warehouse** — tweak item weights after generation to put most
SKUs in the 200–2000 kg range (forklifts required, ergonomic ordering matters
most):
```python
ds = generate(seed=1)
ds.items["weight"] = np.random.default_rng(1).uniform(200, 2000, len(ds.items)).round(1)
```

**Highly skewed demand** — simulate a warehouse where a handful of SKUs drive
almost everything (sharper Pareto, e.g. a 95/5 split).  Open `data_gen.py`,
change the Pareto shape parameter:
```python
picks_60 = (rng.pareto(0.5, n_items) * 8).astype(int) + 1   # was 1.2 → now 0.5
```
Lower shape → heavier tail → more extreme fast/slow split.

**Multi-day demand with trend** — spread orders across many days and observe
how `picks_trend` diverges from 1.0 for fast-growing or declining SKUs:
```python
ds = generate(n_pickruns=2000, n_days=90, seed=7)
# Items with picks_trend > 1.3 are accelerating; < 0.7 are declining
fast_movers = ds.items[ds.items["picks_trend"] > 1.3]
```

**Cold-chain heavy** — most items require refrigeration, amplifying the
cold-last constraint:
```python
ds = generate(seed=3)
# Reassign 80% of items to CHILLER/FREEZER
ds.items["zone_type_1"] = np.where(
    np.random.default_rng(3).random(len(ds.items)) < 0.8,
    np.random.default_rng(4).choice(["CHILLER", "FREEZER"], len(ds.items)),
    "AMBIENT",
)
```
Then compare routes with and without `cold_last=True` — the ordering_fixes
difference quantifies how badly the unconstrained solver violates cold-chain.

**Zone-crossing experiments** — the current DES ignores zone boundaries.  Add
a zone-crossing penalty by checking whether consecutive tiles in a route cross
from one zone to another, and injecting an `env.timeout(penalty_s)` for each
crossing (a natural extension of `_do_route` in `des.py`).

### Plugging in real data

The `Dataset` dataclass is the only contract between the data layer and the
rest of the playground.  To use real data, implement a loader that returns the
same six DataFrames:

```python
# data_loader_real.py  (not in this repo — lives in the main codebase)
def load_from_db(conn, date: str) -> Dataset:
    items        = pd.read_sql("SELECT ...", conn)
    locations    = pd.read_sql("SELECT ...", conn)
    ...
    return Dataset(items, locations, loc_features, item_links, inventory, transactions)
```

Then in `run_demo.py`, swap one line:
```python
# ds = generate(...)       # synthetic
ds = load_from_db(conn, "2024-03-15")   # real
```

Nothing else changes.  The column names in `Dataset` were chosen to match the
real DB schema exactly so this substitution is always a one-liner.

---

## Exploration tasks

### Task 1 — Improve the ACO solver

The `"aco"` solver in `tsp.py` is a working but intentionally basic port.
It gets close to 2-opt quality without any of the standard improvements.
Three concrete things to try, in rough order of difficulty:

**1a. Symmetric pheromone deposit**
Currently, when an ant uses edge (i → j), only `pheromone[i, j]` is updated.
Add `pheromone[j, i] += deposit` as well.  Does it help on the warehouse's
directed graph, or does asymmetry actually capture something useful?

**1b. Elitism**
At the end of each iteration, deposit extra pheromone along the globally best
tour found so far.  This reinforces good solutions more aggressively:

```python
# After the per-ant deposit loop:
elite_deposit = 1.0 / best_cost
for k in range(len(best_route_idx) - 1):
    u = [0] + best_route_idx   # re-insert start node
    pheromone[u[k], u[k+1]] += elite_deposit
```

**1c. Per-ant 2-opt**
After each ant builds its tour, run one or two passes of `two_opt_improve`
before computing the cost used for pheromone deposit.  The existing
`two_opt_improve` function in `tsp.py` is ready to use.

Measure each improvement with the solver comparison in `run_demo.py`.

---

### Task 2 — Dynamic rerouting

The `reroute_on_wait_s` parameter in `run_des()` adds basic rerouting: when a
picker has been waiting at a one-way aisle for longer than the threshold, it
defers all tiles in that aisle to the end of its route and moves on.

This is a simplistic heuristic.  Better approaches to explore:

**2a. Threshold sensitivity**
Run `reroute_on_wait_s` ∈ {5, 10, 20, 30, 60} at 2, 3, and 5 pickers.
Does rerouting help more when contention is high (many pickers) or low?
Plot makespan vs threshold vs n\_pickers.

**2b. Re-solve instead of defer**
Instead of just deferring, re-run the TSP solver on the remaining tiles
(excluding the blocked aisle) and re-attach the blocked aisle at the end.
In `des.py`, the `reroute_on_wait_s` branch currently does a simple list
reorder — replace it with a call to `_apply_solver` for a proper re-route.

**2c. Coordinated dispatch**
A more ambitious extension: add a central dispatcher process that monitors
aisle queues and re-assigns tiles between active pickers to balance load.
This requires a shared "work pool" (a `simpy.Store` of remaining tiles) rather
than per-pickrun routes pre-assigned at the start.

---

### Task 3 — Mixed machinery fleet

The `machine_profile=` parameter lets you simulate different vehicle types.
The current model uses a single profile for all pickers.  Extend this to model
a realistic mixed fleet.

**3a. Profile comparison**
Run `run_demo.py` and compare `HUMAN`, `REACH_TRUCK`, and `COUNTERBALANCE`.
Which machine minimises makespan?  Which minimises average pickrun time?
Why does `COUNTERBALANCE` show zero aisle-wait?

**3b. Assign profiles to pickers**
Modify `run_des()` to accept a list of profiles (one per picker slot) instead
of a single profile.  High-reach items should go to `REACH_TRUCK` pickers;
ground-level heavy items to `COUNTERBALANCE`.

**3c. New profile: electric pallet jack**
Model a slow, narrow-aisle machine:
```python
from des import MachineProfile
pallet_jack = MachineProfile("pallet_jack", speed_m_s=0.8, pick_time_s=3.0)
```
At what picker count does it match the human baseline?

**3d. Zone-restricted routing**
`COUNTERBALANCE` can't enter narrow aisles (`narrow_ok=False`), so it detours.
Currently the TSP route is computed without this constraint.  Add a
`can_use_oneway` flag to `route_pickrun` so the routing graph only uses
bidirectional edges for wide-machine pickruns.  Compare the routed distance vs
the detour-penalty approach.

---

### Task 4 — Cold-chain constraints

The `cold_last=True` flag forces AMBIENT → CHILLER → FREEZER pick ordering so
cold items spend the least time outside refrigeration.  Several follow-on
questions to explore:

**4a. How often does the raw order violate cold-chain?**
Run the solver comparison with and without `cold_last=True` and compare the
`ordering_fixes` column.  Each extra fix represents a cold item that would have
been picked before an ambient one.

**4b. Cost of the constraint**
For most warehouse layouts, cold zones are physically grouped, so enforcing
cold-last often *reduces* total route distance (you stop backtracking into the
cold area).  But in some layouts it could increase distance.  Generate a
warehouse where cold zones are interspersed across all aisles (edit `data_gen.py`
to assign zones randomly rather than by aisle) and measure the distance penalty.

**4c. Dwell-time model in the DES**
Currently the DES ignores temperature — it doesn't track how long cold items
sit in the cart.  Extend `des.py` to compute, for each cold item in a pickrun,
the time from when it was picked to when the pickrun ends (the "dwell time").
This is a better measure of cold-chain compliance than ordering violations alone,
because a correctly-ordered route still has the first CHILLER item in the cart
for the entire FREEZER segment.

**4d. Zone-crossing penalty**
Add an `env.timeout(zone_cross_penalty_s)` in `_do_route` when consecutive
tiles in a route belong to different zones.  Does this change which solver wins?

---

## Key constants

| Constant | Location | Value | Meaning |
|----------|----------|-------|---------|
| `SPEED_M_S` | `tsp.py` | 1.5 m/s | Default picker speed (overridden by `MachineProfile`) |
| `PICK_TIME_S` | `tsp.py` | 4.0 s | Default per-item pick time |
| `HEAVY_KG` | `tsp.py` | 500 kg | Heavy-class weight threshold |
| `FRAGILE_KG` | `tsp.py` | 50 kg | Fragile-class weight threshold |
| `COLD_ZONES` | `tsp.py` | `("AMBIENT","CHILLER","FREEZER")` | Temperature zone order for cold-last routing |
| `UNIT_DIST` | `graph.py` | 2.0 m | Metres per grid unit |
