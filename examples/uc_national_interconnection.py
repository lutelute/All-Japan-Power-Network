#!/usr/bin/env python3
"""National UC with interconnection constraints — 757 real generators.

Loads real power plant data from ``data/{region}_plants.geojson`` (OSM-derived),
applies UC cost defaults, and solves three scenarios:

1. **Per-region UC** — each of the 10 areas solved independently
2. **All-Japan copper plate** — 757 generators in a single MILP, no transmission limits
3. **All-Japan with interconnections** — 757 generators + 9 inter-regional links
   with nodal balance per region and transmission capacity constraints

The comparison between scenarios (2) and (3) quantifies the cost of transmission
congestion across the national grid.

Usage::

    python examples/uc_national_interconnection.py

Requirements:
    - ``data/{region}_plants.geojson`` files (from OSM fetch pipeline)
    - ``data/reference/interconnections.yaml`` (OCCTO interconnection data)
    - PuLP with HiGHS or CBC solver
"""

import json
import os
import sys
import time as _time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.model.generator import Generator
from src.uc.interconnection_loader import InterconnectionLoader
from src.uc.models import DemandProfile, TimeHorizon, UCParameters
from src.uc.solver import solve_uc

# ── Configuration ─────────────────────────────────────────────────────────────

PROJ_ROOT = os.path.join(os.path.dirname(__file__), "..")

NUM_PERIODS = 24
DEMAND_FACTOR = 0.65         # Peak demand as fraction of installed capacity
RESERVE_MARGIN = 0.05        # 5% spinning reserve
MIN_CAPACITY_MW = 5.0        # Generator inclusion threshold
SOLVER_TIME_LIMIT_S = 600    # 10 minutes max per solve
MIP_GAP = 0.01               # 1% optimality gap

REGIONS = [
    "hokkaido", "tohoku", "tokyo", "chubu", "hokuriku",
    "kansai", "chugoku", "shikoku", "kyushu", "okinawa",
]

# Summer weekday demand shape (normalized, peak = 1.0)
DEMAND_SHAPE = np.array([
    0.60, 0.57, 0.55, 0.53, 0.55, 0.60,   # 00-05h: nighttime
    0.68, 0.78, 0.87, 0.93, 0.97, 1.00,   # 06-11h: morning ramp
    0.99, 0.98, 0.96, 0.93, 0.90, 0.86,   # 12-17h: afternoon
    0.82, 0.78, 0.74, 0.70, 0.66, 0.63,   # 18-23h: evening
])

# Fuel type cost mapping (¥/MWh)
FUEL_COST = {
    "coal": 4500, "gas": 7000, "lng": 7000, "oil": 9000,
    "nuclear": 1500, "hydro": 0, "wind": 0, "solar": 0,
    "biomass": 3000, "geothermal": 0, "waste": 5000,
}

# Normalize OSM fuel types to model fuel types
FUEL_MAP = {
    "coal": "coal", "gas": "lng", "lng": "lng", "oil": "oil",
    "nuclear": "nuclear", "hydro": "hydro", "wind": "wind",
    "solar": "solar", "biomass": "biomass", "geothermal": "geothermal",
    "waste": "biomass",
}


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_generators_from_geojson(region):
    """Load generators from a region's GeoJSON and apply UC cost defaults."""
    path = os.path.join(PROJ_ROOT, "data", f"{region}_plants.geojson")
    with open(path) as f:
        data = json.load(f)

    generators = []
    for i, feat in enumerate(data["features"]):
        props = feat["properties"]
        cap = props.get("capacity_mw")
        if not cap or float(cap) < MIN_CAPACITY_MW:
            continue

        cap = float(cap)
        name = props.get("name") or props.get("_display_name") or f"{region}_gen_{i}"
        raw_fuel = (props.get("fuel_type") or props.get("plant:source") or "unknown").lower()
        fuel = FUEL_MAP.get(raw_fuel, "mixed")
        cost = FUEL_COST.get(raw_fuel, 5000)
        osm_id = props.get("osm_id", i)

        is_renewable = fuel in ("hydro", "wind", "solar", "geothermal")
        gen = Generator(
            id=f"{region}_gen_{osm_id}",
            name=name,
            capacity_mw=cap,
            fuel_type=fuel,
            region=region,
            startup_cost=0 if is_renewable else 5000,
            shutdown_cost=0 if is_renewable else 2000,
            min_up_time_h=4 if fuel in ("coal", "nuclear") else 2,
            min_down_time_h=4 if fuel in ("coal", "nuclear") else 2,
            fuel_cost_per_mwh=cost,
            no_load_cost=0 if fuel in ("wind", "solar") else 500,
            labor_cost_per_h=300,
        )
        generators.append(gen)

    return generators


def load_all_generators():
    """Load generators from all 10 regions."""
    region_gens = {}
    all_gens = []
    for region in REGIONS:
        gens = load_generators_from_geojson(region)
        region_gens[region] = gens
        all_gens.extend(gens)
    return region_gens, all_gens


def make_demand(total_cap):
    """Generate a 24h demand profile scaled to installed capacity."""
    return (DEMAND_SHAPE * total_cap * DEMAND_FACTOR).tolist()


# ── UC Solve Helpers ──────────────────────────────────────────────────────────

def solve_region(gens):
    """Solve UC for a single region."""
    total_cap = sum(g.capacity_mw for g in gens)
    demand = make_demand(total_cap)
    params = UCParameters(
        generators=gens,
        demand=DemandProfile(demands=demand),
        time_horizon=TimeHorizon(num_periods=NUM_PERIODS, period_duration_h=1.0),
        reserve_margin=RESERVE_MARGIN,
        solver_time_limit_s=120,
        mip_gap=MIP_GAP,
    )
    return solve_uc(params), total_cap, max(demand)


def solve_national(all_gens, interconnections=None):
    """Solve UC for all generators nationally.

    Args:
        all_gens: List of all generators across all regions.
        interconnections: If provided, enables nodal balance per region
            with transmission capacity constraints. If None, uses a single
            system-wide demand balance (copper plate model).
    """
    total_cap = sum(g.capacity_mw for g in all_gens)
    demand = make_demand(total_cap)
    params = UCParameters(
        generators=all_gens,
        demand=DemandProfile(demands=demand),
        time_horizon=TimeHorizon(num_periods=NUM_PERIODS, period_duration_h=1.0),
        reserve_margin=RESERVE_MARGIN,
        solver_time_limit_s=SOLVER_TIME_LIMIT_S,
        mip_gap=MIP_GAP,
        interconnections=interconnections or [],
    )
    return solve_uc(params), total_cap, max(demand)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    wall_start = _time.monotonic()

    print("=" * 80)
    print("National UC with Interconnection Constraints")
    print(f"  Generators >= {MIN_CAPACITY_MW} MW | {NUM_PERIODS}h horizon | "
          f"{RESERVE_MARGIN*100:.0f}% reserve | MIP gap {MIP_GAP*100:.0f}%")
    print("=" * 80)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    region_gens, all_gens = load_all_generators()
    total_cap = sum(g.capacity_mw for g in all_gens)

    print(f"\nLoaded {len(all_gens)} generators, {total_cap:,.0f} MW across {len(REGIONS)} regions\n")

    # ── Step 2: Per-region UC ─────────────────────────────────────────────
    print("─── (1) Per-Region UC ──────────────────────────────────────────────")
    print(f"\n{'Region':<12} {'Gens':>5} {'Cap(MW)':>10} {'Peak(MW)':>10} "
          f"{'Status':<12} {'Cost':>16} {'Time(s)':>8}")
    print("-" * 82)

    all_ok = True
    for region in REGIONS:
        gens = region_gens[region]
        if not gens:
            print(f"{region:<12}   N/A — no generators above threshold")
            continue

        result, cap, peak = solve_region(gens)
        cost_str = f"¥{result.total_cost:,.0f}" if result.is_optimal else "-"
        print(f"{region:<12} {len(gens):>5} {cap:>10,.0f} {peak:>10,.0f} "
              f"{result.status:<12} {cost_str:>16} {result.solve_time_s:>8.2f}")

        if not result.is_optimal:
            all_ok = False
            for w in result.warnings[:3]:
                print(f"  WARNING: {w}")

    # ── Step 3: All-Japan copper plate ────────────────────────────────────
    print("\n─── (2) All-Japan UC — Copper Plate (no transmission limits) ────────")

    demand = make_demand(total_cap)
    print(f"\n  Generators: {len(all_gens)} | Capacity: {total_cap:,.0f} MW | "
          f"Peak demand: {max(demand):,.0f} MW")
    print("  Solving...")

    result_cp, _, _ = solve_national(all_gens)

    print(f"\n  Status:     {result_cp.status}")
    print(f"  Cost:       ¥{result_cp.total_cost:,.0f}")
    print(f"  Solve time: {result_cp.solve_time_s:.2f}s")

    if result_cp.warnings:
        for w in result_cp.warnings[:3]:
            print(f"  WARNING: {w}")
    if not result_cp.is_optimal:
        all_ok = False

    # ── Step 4: All-Japan with interconnections ───────────────────────────
    print("\n─── (3) All-Japan UC — With Interconnection Constraints ─────────────")

    ic_path = os.path.join(PROJ_ROOT, "data", "reference", "interconnections.yaml")
    loader = InterconnectionLoader()
    interconnections = loader.load(ic_path)

    print(f"\n  Generators: {len(all_gens)} | Interconnections: {len(interconnections)} | "
          f"Peak demand: {max(demand):,.0f} MW")
    print("  Solving...")

    result_ic, _, _ = solve_national(all_gens, interconnections)

    print(f"\n  Status:     {result_ic.status}")
    print(f"  Cost:       ¥{result_ic.total_cost:,.0f}")
    print(f"  Solve time: {result_ic.solve_time_s:.2f}s")

    if result_ic.warnings:
        for w in result_ic.warnings[:3]:
            print(f"  WARNING: {w}")
    if not result_ic.is_optimal:
        all_ok = False

    if result_ic.interconnection_flows:
        print(f"\n  {'Interconnection':<42} {'Max Flow':>10} {'Capacity':>10} {'Util':>6}")
        print("  " + "-" * 72)
        for ic in interconnections:
            flow = next((f for f in result_ic.interconnection_flows
                         if f.interconnection_id == ic.id), None)
            if flow:
                max_flow = max(abs(v) for v in flow.flow_mw)
                util = max_flow / ic.capacity_mw * 100
                sat = " SAT" if max_flow >= ic.capacity_mw * 0.999 else ""
                print(f"  {ic.name_en:<42} {max_flow:>8,.0f} MW {ic.capacity_mw:>8,.0f} MW"
                      f" {util:>5.1f}%{sat}")

    # ── Step 5: Comparison ────────────────────────────────────────────────
    print("\n─── Comparison ─────────────────────────────────────────────────────")
    if result_cp.is_optimal and result_ic.is_optimal:
        delta = result_ic.total_cost - result_cp.total_cost
        pct = delta / result_cp.total_cost * 100 if result_cp.total_cost > 0 else 0
        print(f"  Copper plate:       ¥{result_cp.total_cost:>16,.0f}  ({result_cp.solve_time_s:.1f}s)")
        print(f"  With IC constraints: ¥{result_ic.total_cost:>15,.0f}  ({result_ic.solve_time_s:.1f}s)")
        print(f"  Congestion cost:    ¥{delta:>16,.0f}  ({pct:+.2f}%)")

    wall_elapsed = _time.monotonic() - wall_start

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    if all_ok:
        print(f"ALL SCENARIOS OPTIMAL — {len(all_gens)} generators feasible across all tests")
        print(f"  Per-region (×{len(REGIONS)}) + All-Japan copper plate + All-Japan with IC")
    else:
        print("SOME SCENARIOS FAILED — check warnings above")
    print(f"Total wall time: {wall_elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
