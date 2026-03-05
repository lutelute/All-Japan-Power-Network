#!/usr/bin/env python3
"""Advanced UC demo: thermal + pumped-storage hydro + battery with adaptive solver.

Demonstrates the full capabilities of the adaptive unit commitment solver
including:

- **Thermal generators**: Coal, LNG, oil with fuel costs, startup/shutdown costs,
  ramp rate limits, and minimum up/down time constraints.
- **Pumped-storage hydro**: A large pumped-hydro plant with charge/discharge
  cycling, round-trip efficiency losses, and state-of-charge tracking.
- **Battery storage**: A utility-scale battery with high charge/discharge rates,
  SOC bounds, and terminal SOC requirements.
- **Adaptive solving**: Hardware detection, solver tier selection, and
  detailed performance reporting.

The script creates a mixed fleet across two regions (Tokyo, Chubu), solves
a 24-hour day-ahead unit commitment using ``solve_adaptive()``, and prints
comprehensive results including hardware profile, solver tier, generator
schedules, storage SOC profiles, and cost breakdown.

Usage::

    python examples/uc_advanced_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.generator import Generator
from src.uc.models import UCParameters, TimeHorizon, DemandProfile
from src.uc.adaptive_solver import AdaptiveUCResult, solve_adaptive
from src.uc.hardware_detector import detect_hardware

# ── Constants ─────────────────────────────────────────────────────
NUM_PERIODS = 24
RESERVE_MARGIN = 0.05


def make_generators():
    """Create a mixed fleet: thermal + pumped-storage hydro + battery.

    Returns a list of 10 generators across Tokyo and Chubu regions:
    - 6 thermal generators (nuclear, coal, LNG, oil)
    - 1 pumped-storage hydro unit (揚水発電所)
    - 1 conventional hydro unit
    - 1 battery storage unit
    - 1 wind generator
    """
    gens = [
        # ── Tokyo (関東) — Thermal generators ──
        Generator(
            id="tokyo_nuc_01", name="柏崎刈羽原発", capacity_mw=1100,
            fuel_type="nuclear", region="tokyo",
            startup_cost=50000, shutdown_cost=20000,
            min_up_time_h=12, min_down_time_h=12,
            fuel_cost_per_mwh=1500, no_load_cost=800, labor_cost_per_h=500,
        ),
        Generator(
            id="tokyo_coal_01", name="磯子火力(石炭)", capacity_mw=600,
            fuel_type="coal", region="tokyo",
            startup_cost=8000, shutdown_cost=3000,
            min_up_time_h=6, min_down_time_h=4,
            ramp_up_mw_per_h=120, ramp_down_mw_per_h=120,
            fuel_cost_per_mwh=4500, no_load_cost=600, labor_cost_per_h=400,
        ),
        Generator(
            id="tokyo_lng_01", name="富津LNG-1", capacity_mw=500,
            fuel_type="lng", region="tokyo",
            startup_cost=5000, shutdown_cost=2000,
            min_up_time_h=3, min_down_time_h=2,
            ramp_up_mw_per_h=200, ramp_down_mw_per_h=200,
            fuel_cost_per_mwh=7000, no_load_cost=400, labor_cost_per_h=300,
        ),
        Generator(
            id="tokyo_oil_01", name="横須賀石油火力", capacity_mw=350,
            fuel_type="oil", region="tokyo",
            startup_cost=3000, shutdown_cost=1500,
            min_up_time_h=2, min_down_time_h=2,
            ramp_up_mw_per_h=250, ramp_down_mw_per_h=250,
            fuel_cost_per_mwh=9000, no_load_cost=300, labor_cost_per_h=200,
        ),
        # ── Tokyo — Storage: Battery ──
        Generator(
            id="tokyo_bat_01", name="東京大規模蓄電池", capacity_mw=200,
            fuel_type="hydro", region="tokyo",
            startup_cost=100, shutdown_cost=50,
            min_up_time_h=1, min_down_time_h=1,
            ramp_up_mw_per_h=200, ramp_down_mw_per_h=200,
            fuel_cost_per_mwh=0, no_load_cost=50, labor_cost_per_h=30,
            storage_capacity_mwh=800,
            charge_rate_mw=200,
            discharge_rate_mw=200,
            charge_efficiency=0.92,
            discharge_efficiency=0.95,
            initial_soc_fraction=0.5,
            min_terminal_soc_fraction=0.3,
        ),
        # ── Chubu (中部) — Thermal generators ──
        Generator(
            id="chubu_coal_01", name="碧南火力(石炭)", capacity_mw=700,
            fuel_type="coal", region="chubu",
            startup_cost=9000, shutdown_cost=3500,
            min_up_time_h=6, min_down_time_h=4,
            ramp_up_mw_per_h=140, ramp_down_mw_per_h=140,
            fuel_cost_per_mwh=4200, no_load_cost=650, labor_cost_per_h=400,
        ),
        Generator(
            id="chubu_lng_01", name="知多LNG", capacity_mw=400,
            fuel_type="lng", region="chubu",
            startup_cost=4500, shutdown_cost=1800,
            min_up_time_h=3, min_down_time_h=2,
            ramp_up_mw_per_h=180, ramp_down_mw_per_h=180,
            fuel_cost_per_mwh=6800, no_load_cost=350, labor_cost_per_h=280,
        ),
        # ── Chubu — Conventional hydro ──
        Generator(
            id="chubu_hydro_01", name="奥矢作水力", capacity_mw=200,
            fuel_type="hydro", region="chubu",
            startup_cost=500, shutdown_cost=200,
            min_up_time_h=1, min_down_time_h=1,
            ramp_up_mw_per_h=200, ramp_down_mw_per_h=200,
            fuel_cost_per_mwh=0, no_load_cost=100, labor_cost_per_h=100,
        ),
        # ── Chubu — Storage: Pumped-storage hydro ──
        Generator(
            id="chubu_pump_01", name="奥美濃揚水発電所", capacity_mw=300,
            fuel_type="pumped_hydro", region="chubu",
            startup_cost=800, shutdown_cost=300,
            min_up_time_h=1, min_down_time_h=1,
            ramp_up_mw_per_h=300, ramp_down_mw_per_h=300,
            fuel_cost_per_mwh=0, no_load_cost=150, labor_cost_per_h=80,
            storage_capacity_mwh=1800,
            charge_rate_mw=300,
            discharge_rate_mw=300,
            charge_efficiency=0.85,
            discharge_efficiency=0.90,
            initial_soc_fraction=0.6,
            min_terminal_soc_fraction=0.4,
        ),
        # ── Chubu — Wind ──
        Generator(
            id="chubu_wind_01", name="浜松ウィンドファーム", capacity_mw=100,
            fuel_type="wind", region="chubu",
            startup_cost=0, shutdown_cost=0,
            min_up_time_h=1, min_down_time_h=1,
            fuel_cost_per_mwh=0, no_load_cost=0, labor_cost_per_h=30,
        ),
    ]
    return gens


def make_demand_profile():
    """Create a realistic 24h demand curve (summer weekday).

    Scaled for a 2-region (Tokyo + Chubu) fleet with ~4,450 MW total
    capacity, targeting ~65% peak utilization.
    """
    demand = [
        1800, 1700, 1650, 1600, 1650, 1800,   # 0-5h:  nighttime low
        2100, 2500, 2800, 3000, 3150, 3250,   # 6-11h: morning ramp
        3300, 3350, 3250, 3100, 2950, 2800,   # 12-17h: afternoon peak & decline
        2650, 2500, 2350, 2200, 2050, 1900,   # 18-23h: evening decline
    ]
    return demand


def print_hardware_profile(profile):
    """Print detected hardware capabilities."""
    print("=" * 70)
    print("  HARDWARE PROFILE")
    print("=" * 70)
    print(f"  OS / Architecture:     {profile.os_name} / {profile.architecture}")
    print(f"  Physical CPU cores:    {profile.physical_cores}")
    print(f"  Logical CPU cores:     {profile.logical_cores}")
    print(f"  Available RAM:         {profile.available_ram_gb:.1f} GB")
    print(f"  Total RAM:             {profile.total_ram_gb:.1f} GB")
    print(f"  Available solvers:     {', '.join(profile.available_solvers) or 'none detected'}")
    print()


def print_solver_config(adaptive_result):
    """Print adaptive solver configuration and tier selection."""
    config = adaptive_result.solver_config
    print("=" * 70)
    print("  ADAPTIVE SOLVER CONFIGURATION")
    print("=" * 70)
    print(f"  Solver Tier:           {config.tier.value.upper()}")
    print(f"  Solver Backend:        {config.solver_name}")
    print(f"  MIP Gap Tolerance:     {config.mip_gap:.1%}")
    print(f"  Time Limit:            {config.time_limit_s:.0f}s")
    print(f"  Threads:               {config.threads}")
    print(f"  Decomposition:         {'Yes (' + config.decomposition_strategy + ')' if config.use_decomposition else 'No'}")
    print(f"  LP Relaxation:         {'Yes' if config.use_lp_relaxation else 'No'}")
    print(f"  Description:           {config.description}")
    print()

    if adaptive_result.degradation_history:
        print("  Degradation History:")
        for entry in adaptive_result.degradation_history:
            print(f"    - {entry}")
        print()


def print_fleet_summary(gens):
    """Print generator fleet summary table."""
    print("=" * 70)
    print("  GENERATOR FLEET SUMMARY")
    print("=" * 70)
    print(f"  {'ID':<20} {'Name':<18} {'Type':<12} {'Cap(MW)':>8} {'Region':<8} {'Storage'}")
    print("  " + "-" * 80)

    total_cap = 0
    storage_count = 0
    for g in gens:
        storage_info = ""
        if g.is_storage:
            storage_info = f"{g.storage_capacity_mwh:.0f} MWh"
            storage_count += 1
        ramp_info = ""
        if g.ramp_up_mw_per_h is not None:
            ramp_info = f" (ramp: {g.ramp_up_mw_per_h:.0f} MW/h)"
        print(
            f"  {g.id:<20} {g.name:<18} {g.fuel_type:<12} "
            f"{g.capacity_mw:>8.0f} {g.region:<8} {storage_info}{ramp_info}"
        )
        total_cap += g.capacity_mw

    print("  " + "-" * 80)
    print(f"  Total: {len(gens)} generators, {total_cap:,.0f} MW capacity, "
          f"{storage_count} storage units")
    print()


def print_uc_results(adaptive_result, gens, demand):
    """Print detailed UC results including schedules and storage SOC."""
    result = adaptive_result.result
    gen_map = {g.id: g for g in gens}

    print("=" * 70)
    print("  UC SOLVE RESULTS")
    print("=" * 70)
    print(f"  Solver Status:         {result.status}")
    print(f"  Total Cost:            ¥{result.total_cost:,.0f}")
    print(f"  Solve Time:            {result.solve_time_s:.2f}s")
    print(f"  Adaptive Total Time:   {adaptive_result.total_time_s:.2f}s")
    print(f"  Tier Used:             {adaptive_result.tier_used.value.upper()}")
    print(f"  Generators Scheduled:  {result.num_generators}")
    if result.gap is not None:
        print(f"  MIP Gap:               {result.gap:.4%}")
    print()

    if result.warnings:
        print("  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")
        print()

    # ── Generator schedule table ──
    print("  " + "-" * 68)
    print(f"  {'Generator':<22} {'On(h)':>6} {'Output(MWh)':>12} {'Cost(¥)':>14} {'Startups':>8}")
    print("  " + "-" * 68)

    total_energy = 0
    total_cost_check = 0
    for sched in result.schedules:
        g = gen_map.get(sched.generator_id)
        if g is None:
            continue
        hours_on = sum(sched.commitment)
        energy = sum(sched.power_output_mw)
        total_energy += energy
        total_cost_check += sched.total_cost
        print(
            f"  {g.name:<22} {hours_on:>6} {energy:>12,.0f} "
            f"¥{sched.total_cost:>13,.0f} {sched.num_startups:>8}"
        )

    print("  " + "-" * 68)
    total_demand = sum(demand)
    print(f"  Total energy generated:   {total_energy:>10,.0f} MWh")
    print(f"  Total demand:             {total_demand:>10,.0f} MWh")
    print(f"  Peak demand:              {max(demand):>10,.0f} MW")
    print()

    # ── Storage SOC profiles ──
    storage_scheds = [
        (sched, gen_map[sched.generator_id])
        for sched in result.schedules
        if sched.generator_id in gen_map and gen_map[sched.generator_id].is_storage
    ]

    if storage_scheds:
        print("=" * 70)
        print("  STORAGE STATE-OF-CHARGE (SOC) PROFILES")
        print("=" * 70)

        for sched, g in storage_scheds:
            print(f"\n  {g.name} ({g.id}) — {g.storage_capacity_mwh:.0f} MWh capacity")
            print(f"  Charge rate: {g.charge_rate_mw} MW, "
                  f"Discharge rate: {g.discharge_rate_mw} MW")
            print(f"  Efficiency: charge={g.charge_efficiency:.0%}, "
                  f"discharge={g.discharge_efficiency:.0%}")

            if sched.soc_mwh:
                print(f"\n  {'Hour':>6} {'SOC(MWh)':>10} {'SOC(%)':>8} "
                      f"{'Charge(MW)':>12} {'Discharge(MW)':>14} {'Net(MW)':>10}")
                print("  " + "-" * 64)
                for t in range(min(NUM_PERIODS, len(sched.soc_mwh))):
                    soc = sched.soc_mwh[t]
                    soc_pct = (soc / g.storage_capacity_mwh * 100
                               if g.storage_capacity_mwh > 0 else 0)
                    charge = sched.charge_mw[t] if t < len(sched.charge_mw) else 0
                    discharge = sched.discharge_mw[t] if t < len(sched.discharge_mw) else 0
                    net = discharge - charge
                    print(
                        f"  {t:>6} {soc:>10.1f} {soc_pct:>7.1f}% "
                        f"{charge:>12.1f} {discharge:>14.1f} {net:>10.1f}"
                    )
            else:
                print("  (No SOC data available)")

        print()

    # ── Cost breakdown by category ──
    print("=" * 70)
    print("  COST BREAKDOWN")
    print("=" * 70)
    total_fuel = sum(s.fuel_cost for s in result.schedules)
    total_startup = sum(s.startup_cost for s in result.schedules)
    total_shutdown = sum(s.shutdown_cost for s in result.schedules)
    total_noload = sum(s.no_load_cost for s in result.schedules)
    print(f"  Fuel cost:      ¥{total_fuel:>14,.0f}  ({total_fuel / result.total_cost * 100:.1f}%)" if result.total_cost > 0 else f"  Fuel cost:      ¥{total_fuel:>14,.0f}")
    print(f"  No-load cost:   ¥{total_noload:>14,.0f}  ({total_noload / result.total_cost * 100:.1f}%)" if result.total_cost > 0 else f"  No-load cost:   ¥{total_noload:>14,.0f}")
    print(f"  Startup cost:   ¥{total_startup:>14,.0f}  ({total_startup / result.total_cost * 100:.1f}%)" if result.total_cost > 0 else f"  Startup cost:   ¥{total_startup:>14,.0f}")
    print(f"  Shutdown cost:  ¥{total_shutdown:>14,.0f}  ({total_shutdown / result.total_cost * 100:.1f}%)" if result.total_cost > 0 else f"  Shutdown cost:  ¥{total_shutdown:>14,.0f}")
    print(f"  {'':>16}  {'─' * 18}")
    print(f"  TOTAL:          ¥{result.total_cost:>14,.0f}")
    print()

    # ── Ramp rate compliance check ──
    print("=" * 70)
    print("  RAMP RATE COMPLIANCE CHECK")
    print("=" * 70)
    violations = 0
    for sched in result.schedules:
        g = gen_map.get(sched.generator_id)
        if g is None or g.ramp_up_mw_per_h is None:
            continue
        for t in range(1, len(sched.power_output_mw)):
            # Ramp constraints only apply between consecutive committed periods
            if sched.commitment[t] == 0 or sched.commitment[t - 1] == 0:
                continue
            delta = sched.power_output_mw[t] - sched.power_output_mw[t - 1]
            if delta > g.ramp_up_mw_per_h + 0.1:
                violations += 1
            if -delta > g.ramp_down_mw_per_h + 0.1:
                violations += 1

    if violations == 0:
        print("  All generators comply with ramp rate limits.")
    else:
        print(f"  WARNING: {violations} ramp rate violation(s) detected.")
        if adaptive_result.solver_config and adaptive_result.solver_config.use_lp_relaxation:
            print("  (Expected with LP relaxation — rounded solution is approximate)")
    print()


def main():
    """Run the advanced UC demo with adaptive solver."""
    print()
    print("=" * 70)
    print("  ADVANCED UC SOLVER DEMO")
    print("  Thermal + Pumped-Storage Hydro + Battery | Adaptive Solver")
    print("=" * 70)
    print()

    # ── Step 1: Detect hardware ──
    print("STEP 1: Detecting hardware capabilities...")
    profile = detect_hardware()
    print_hardware_profile(profile)

    # ── Step 2: Create generator fleet ──
    print("STEP 2: Creating generator fleet...")
    gens = make_generators()
    print_fleet_summary(gens)

    # ── Step 3: Create demand profile ──
    print("STEP 3: Creating 24-hour demand profile...")
    demand = make_demand_profile()
    print(f"  Peak demand:  {max(demand):,.0f} MW (hour {demand.index(max(demand))})")
    print(f"  Min demand:   {min(demand):,.0f} MW (hour {demand.index(min(demand))})")
    print(f"  Total energy: {sum(demand):,.0f} MWh")
    print()

    # ── Step 4: Solve with adaptive solver ──
    print("STEP 4: Solving UC with adaptive solver (solve_adaptive)...")
    print()

    th = TimeHorizon(num_periods=NUM_PERIODS, period_duration_h=1.0)
    dp = DemandProfile(demands=demand)
    params = UCParameters(
        generators=gens,
        demand=dp,
        time_horizon=th,
        reserve_margin=RESERVE_MARGIN,
    )

    adaptive_result = solve_adaptive(params)

    # ── Step 5: Print solver configuration ──
    print()
    print("STEP 5: Solver configuration and tier selection")
    print_solver_config(adaptive_result)

    # ── Step 6: Print detailed results ──
    print("STEP 6: Detailed results")
    print_uc_results(adaptive_result, gens, demand)

    # ── Final status ──
    print("=" * 70)
    print(f"  Solver Status: {adaptive_result.result.status}")
    print(f"  Tier Used: {adaptive_result.tier_used.value.upper()}")
    print(f"  Total Cost: ¥{adaptive_result.result.total_cost:,.0f}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
