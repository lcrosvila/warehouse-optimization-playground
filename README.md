# Warehouse Optimisation Playground

A self-contained Python sandbox for experimenting with **TSP** (Travelling
Salesman Problem) routing and **DES** (Discrete-Event Simulation) for
warehouse pick-route optimisation.

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
```

Expected output: a TSP comparison table (NN vs NN + 2-opt) and a picker-sweep
table showing makespan vs staffing level.

---

## Background

In a warehouse, a *pickrun* is one customer order: a picker walks a route and
collects a set of items from their shelf locations.  Two problems drive most
of the optimisation work:

1. **TSP** — in what order should the picker visit the locations to minimise
   total walking distance?
2. **DES** — given several concurrent pickers, how do shared resources (aisles,
   picker pool) create contention, and what is the overall *makespan*?

The warehouse is modelled as a rectangular grid with *one-way aisles* (odd-x
columns are traversed south → north only).  One-way aisles prevent collisions
but create queuing when multiple pickers want the same aisle at the same time.

Pick-ordering constraints apply for ergonomic and damage-prevention reasons:

| Segment | Class    | Rule |
|---------|----------|------|
| 1st     | Heavy    | weight ≥ 500 kg — picked first (at the bottom of the cart) |
| 2nd     | Normal   | all other items |
| 3rd     | Fragile  | weight ≤ 50 kg — picked last (on top) |

---

## Files

| File | Role |
|------|------|
| `graph.py` | Directed routing graph: one-way aisles, depot, Dijkstra distances |
| `data_gen.py` | Synthetic data generator — produces the same DataFrame schema as the real DB |
| `tsp.py` | TSP solvers: nearest-neighbor baseline + 2-opt improvement |
| `des.py` | SimPy DES: picker pool + one-way aisle contention model |
| `run_demo.py` | End-to-end demo: generate → route (2 methods) → simulate (picker sweep) |

All files are self-contained; they do **not** import from the rest of this
repository.

---

## Data schema

`data_gen.generate()` returns a `Dataset` with six DataFrames.  The column
names match the real database loader exactly, so swapping in real data
requires only replacing the `generate()` call — the rest of the code is
unchanged.

| DataFrame | Key columns |
|-----------|-------------|
| `items` | `item`, `weight`, `box_qty`, `picks_30`, `picks_60`, `picks_trend`, `zone_type_1` |
| `locations` | `location_id`, `x_coord`, `y_coord`, `weight_limit`, `zone_type_1`, `module_id` |
| `loc_features` | `location_id`, `time_to_start_point` |
| `inventory` | `item`, `location_id`, `qty` |
| `transactions` | `pickrun_no`, `order_no`, `item`, `location_id`, `qty`, `day`, `timestamp` |
| `item_links` | `item`, `community_id`, `community_priority` |

---

## Experiment ideas

### TSP
- **Baseline established**: nearest-neighbor heuristic in `tsp.nearest_neighbor()`.
- **First experiment**: 2-opt is already implemented — run the demo and observe the improvement.
- **Next steps**:
  - Implement **or-opt**: move single tiles to a better position instead of reversing segments.
  - Implement **3-opt** and compare the improvement-vs-runtime trade-off.
  - Try a **zone-aware** strategy: cluster picks by module or aisle before NN.
  - Study how the heavy/fragile segmentation affects overall route length.

### DES
- Sweep `n_pickers` and plot makespan to find the throughput knee.
- Set `model_aisle_contention=False` and measure how much one-way aisles add.
- Add a **replenishment queue**: a separate worker restocks slots when a picker
  depletes one; each restock takes a fixed service time.
- Model **picker fatigue**: reduce `SPEED_M_S` by a small factor after every N picks
  within a shift.
- Model **zone-crossing penalties**: e.g. entering the CHILLER zone adds a fixed delay.

### Connecting to real data

Replace `data_gen.generate()` with any function that returns a `Dataset` with
the column schemas above.  Nothing else in the playground needs to change.

The main repository's `data_loader.load_from_db()` returns an identical
`Dataset` (from `data.py`); to use it here, copy the six DataFrames into
the playground `Dataset` dataclass.

---

## Key constants (in `tsp.py` and `des.py`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `SPEED_M_S` | 1.5 m/s | Picker walking speed |
| `PICK_TIME_S` | 4.0 s | Time to pick one item from its slot |
| `HEAVY_KG` | 500 kg | Weight threshold for "heavy" class |
| `FRAGILE_KG` | 50 kg | Weight threshold for "fragile" class |
| `UNIT_DIST` | 2.0 m | Metres per grid unit in the routing graph |
