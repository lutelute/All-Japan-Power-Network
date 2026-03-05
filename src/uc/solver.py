"""Core MILP-based unit commitment (UC) solver.

Formulates and solves a mixed-integer linear programming (MILP) unit
commitment problem using PuLP.  The solver determines optimal generator
on/off schedules over a configurable time horizon to minimise total
operating cost while satisfying demand and operational constraints.

Usage::

    from src.uc.solver import solve_uc
    from src.uc.models import UCParameters, TimeHorizon, DemandProfile
    from src.model.generator import Generator

    gens = [Generator(id='g1', name='G1', capacity_mw=200,
                      fuel_type='coal', fuel_cost_per_mwh=30)]
    th = TimeHorizon(num_periods=24, period_duration_h=1.0)
    dp = DemandProfile(demands=[100.0] * 24)
    params = UCParameters(generators=gens, demand=dp, time_horizon=th)
    result = solve_uc(params)
    print(result.status, result.total_cost)
"""

import time
from typing import Dict, List, Optional, Tuple

import pulp

from src.model.generator import Generator
from src.uc.constraints import (
    add_capacity_bounds_constraints,
    add_demand_balance_constraints,
    add_maintenance_constraints,
    add_min_downtime_constraints,
    add_min_uptime_constraints,
    add_nodal_balance_constraints,
    add_ramp_constraints,
    add_reserve_margin_constraints,
    add_startup_shutdown_logic,
    add_storage_soc_constraints,
    add_transmission_capacity_constraints,
)
from src.uc.models import (
    GeneratorSchedule,
    Interconnection,
    InterconnectionFlow,
    UCParameters,
    UCResult,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# PuLP status code to human-readable string mapping
_STATUS_MAP = {
    pulp.constants.LpStatusOptimal: "Optimal",
    pulp.constants.LpStatusNotSolved: "Not Solved",
    pulp.constants.LpStatusInfeasible: "Infeasible",
    pulp.constants.LpStatusUnbounded: "Unbounded",
    pulp.constants.LpStatusUndefined: "Undefined",
}


def solve_uc(params: UCParameters) -> UCResult:
    """Solve a unit commitment optimisation problem.

    Formulates a MILP using PuLP with binary commitment, startup, and
    shutdown variables plus continuous power output variables.  The
    objective minimises total operating cost (fuel, no-load, labour,
    startup, and shutdown costs).  Constraints are added via modular
    builders from ``src.uc.constraints``.

    Args:
        params: UC problem specification including generators, demand
            profile, time horizon, and solver configuration.

    Returns:
        UCResult with per-generator schedules, total cost, solver
        status, and solve time.  On infeasibility, diagnostic
        warnings identify which timesteps have insufficient capacity.
    """
    result = UCResult()

    # --- Validate inputs ---------------------------------------------------
    if not params.generators:
        result.status = "Infeasible"
        result.warnings.append("No generators provided")
        logger.warning("UC solve aborted: no generators provided")
        return result

    if params.demand is None or params.time_horizon is None:
        result.status = "Not Solved"
        result.warnings.append("Demand profile and time horizon are required")
        logger.warning("UC solve aborted: missing demand or time horizon")
        return result

    generators = params.generators
    timesteps = params.time_horizon.period_indices
    demand = params.demand.demands

    logger.info(
        "UC problem: %d generators, %d timesteps, peak demand=%.1f MW",
        len(generators),
        len(timesteps),
        params.demand.peak_demand,
    )

    # --- Pre-solve feasibility check ---------------------------------------
    _preflight_check(generators, timesteps, demand, result)

    # --- Create model ------------------------------------------------------
    model = pulp.LpProblem("UnitCommitment", pulp.LpMinimize)

    # --- Decision variables ------------------------------------------------
    gen_ids = [g.id for g in generators]
    indices = [(g_id, t) for g_id in gen_ids for t in timesteps]

    # Identify storage generators
    storage_ids = {g.id for g in generators if g.is_storage}
    storage_map = {g.id: g for g in generators if g.is_storage}

    u = pulp.LpVariable.dicts("u", indices, cat="Binary")
    v = pulp.LpVariable.dicts("v", indices, cat="Binary")
    w = pulp.LpVariable.dicts("w", indices, cat="Binary")

    # Split p variable creation: non-storage (lowBound=0), storage (lowBound=None)
    non_storage_p_indices = [
        (g_id, t) for g_id in gen_ids if g_id not in storage_ids for t in timesteps
    ]
    storage_p_indices = [
        (g_id, t) for g_id in gen_ids if g_id in storage_ids for t in timesteps
    ]

    p = {}
    if non_storage_p_indices:
        p.update(
            pulp.LpVariable.dicts("p", non_storage_p_indices, lowBound=0)
        )
    if storage_p_indices:
        p.update(
            pulp.LpVariable.dicts("p", storage_p_indices, lowBound=None)
        )

    # Storage-specific variables
    if storage_ids:
        storage_indices = [
            (g_id, t) for g_id in storage_ids for t in timesteps
        ]
        p_ch = pulp.LpVariable.dicts("p_ch", storage_indices, lowBound=0)
        p_dis = pulp.LpVariable.dicts("p_dis", storage_indices, lowBound=0)
        z_ch = pulp.LpVariable.dicts("z_ch", storage_indices, cat="Binary")
        soc = pulp.LpVariable.dicts("soc", storage_indices, lowBound=0)

        # Set SOC upper bounds per generator
        for g_id, t in storage_indices:
            soc[(g_id, t)].upBound = storage_map[g_id].storage_capacity_mwh

        logger.info(
            "Created %d storage variables (%d storage generators × %d timesteps × 4: p_ch, p_dis, z_ch, soc)",
            len(storage_indices) * 4,
            len(storage_ids),
            len(timesteps),
        )
    else:
        p_ch = {}
        p_dis = {}
        z_ch = {}
        soc = {}

    # Interconnection flow variables
    interconnections = params.interconnections
    f: Dict[Tuple[str, int], pulp.LpVariable] = {}
    if interconnections:
        ic_indices = [
            (ic.id, t) for ic in interconnections for t in timesteps
        ]
        f = pulp.LpVariable.dicts(
            "f", ic_indices, lowBound=None, cat="Continuous"
        )
        logger.info(
            "Created %d flow variables (%d interconnections × %d timesteps)",
            len(ic_indices),
            len(interconnections),
            len(timesteps),
        )

    logger.info(
        "Created %d decision variables (%d generators × %d timesteps × 4)",
        len(indices) * 4,
        len(gen_ids),
        len(timesteps),
    )

    # --- Objective function ------------------------------------------------
    _build_objective(model, u, p, v, w, generators, timesteps)

    # --- Constraints -------------------------------------------------------
    _add_all_constraints(
        model, u, p, v, w, generators, timesteps, demand, params.reserve_margin,
        p_ch=p_ch, p_dis=p_dis, z_ch=z_ch, soc=soc,
        period_duration_h=params.time_horizon.period_duration_h,
        interconnections=interconnections, f=f,
    )

    # --- Select and run solver ---------------------------------------------
    solver = _select_solver(params)
    start_time = time.monotonic()

    logger.info("Solving UC problem...")
    status_code = model.solve(solver)
    elapsed = time.monotonic() - start_time

    result.solve_time_s = elapsed
    result.status = _STATUS_MAP.get(status_code, "Unknown")

    logger.info(
        "Solver finished: status=%s, time=%.2fs",
        result.status,
        elapsed,
    )

    # --- Extract results ---------------------------------------------------
    if result.status == "Optimal":
        _extract_solution(
            result, model, u, p, v, w, generators, timesteps,
            p_ch=p_ch, p_dis=p_dis, soc=soc, storage_ids=storage_ids,
            interconnections=interconnections, f=f,
        )
    elif result.status == "Infeasible":
        _diagnose_infeasibility(result, generators, timesteps, demand)

    logger.info("UC result summary: %s", result.summary)
    return result


def _preflight_check(
    generators: List[Generator],
    timesteps: List[int],
    demand: List[float],
    result: UCResult,
) -> None:
    """Run quick feasibility checks before model construction.

    Checks that total available capacity can meet demand at each
    timestep, accounting for maintenance windows.  Issues are
    recorded as warnings but do not prevent solve (the solver
    will report infeasibility definitively).
    """
    for idx, t in enumerate(timesteps):
        available_cap = 0.0
        for g in generators:
            in_maintenance = any(
                start_h <= t < end_h for start_h, end_h in g.maintenance_windows
            )
            if not in_maintenance:
                available_cap += g.capacity_mw
        if available_cap < demand[idx]:
            shortfall = demand[idx] - available_cap
            msg = (
                f"Timestep {t}: demand ({demand[idx]:.1f} MW) exceeds "
                f"available capacity ({available_cap:.1f} MW) by "
                f"{shortfall:.1f} MW"
            )
            result.warnings.append(msg)
            logger.warning(msg)


def _build_objective(
    model: pulp.LpProblem,
    u: Dict[Tuple[str, int], pulp.LpVariable],
    p: Dict[Tuple[str, int], pulp.LpVariable],
    v: Dict[Tuple[str, int], pulp.LpVariable],
    w: Dict[Tuple[str, int], pulp.LpVariable],
    generators: List[Generator],
    timesteps: List[int],
) -> None:
    """Build the cost-minimisation objective function.

    Total cost = Σ_g Σ_t [ fuel_cost[g] * p[g,t]
                          + (no_load_cost[g] + labor_cost[g]) * u[g,t]
                          + startup_cost[g] * v[g,t]
                          + shutdown_cost[g] * w[g,t] ]
    """
    objective = pulp.lpSum(
        g.fuel_cost_per_mwh * p[(g.id, t)]
        + (g.no_load_cost + g.labor_cost_per_h) * u[(g.id, t)]
        + g.startup_cost * v[(g.id, t)]
        + g.shutdown_cost * w[(g.id, t)]
        for g in generators
        for t in timesteps
    )
    model += objective
    logger.info("Objective function built: minimise total operating cost")


def _add_all_constraints(
    model: pulp.LpProblem,
    u: Dict[Tuple[str, int], pulp.LpVariable],
    p: Dict[Tuple[str, int], pulp.LpVariable],
    v: Dict[Tuple[str, int], pulp.LpVariable],
    w: Dict[Tuple[str, int], pulp.LpVariable],
    generators: List[Generator],
    timesteps: List[int],
    demand: List[float],
    reserve_margin: float,
    *,
    p_ch: Optional[Dict[Tuple[str, int], pulp.LpVariable]] = None,
    p_dis: Optional[Dict[Tuple[str, int], pulp.LpVariable]] = None,
    z_ch: Optional[Dict[Tuple[str, int], pulp.LpVariable]] = None,
    soc: Optional[Dict[Tuple[str, int], pulp.LpVariable]] = None,
    period_duration_h: float = 1.0,
    interconnections: Optional[List[Interconnection]] = None,
    f: Optional[Dict[Tuple[str, int], pulp.LpVariable]] = None,
) -> None:
    """Add all constraint classes to the model.

    When interconnections are present, uses per-region nodal balance
    constraints instead of system-wide demand balance, and adds
    transmission capacity constraints on flow variables.
    """
    if interconnections and f:
        # Nodal balance replaces system-wide demand balance
        regional_demand = _split_demand_by_region(generators, demand)
        add_nodal_balance_constraints(
            model, p, f, generators, interconnections, timesteps,
            regional_demand,
        )
        add_transmission_capacity_constraints(
            model, f, interconnections, timesteps,
        )
    else:
        add_demand_balance_constraints(model, p, generators, timesteps, demand)
    add_capacity_bounds_constraints(model, u, p, generators, timesteps)
    add_startup_shutdown_logic(model, u, v, w, generators, timesteps)
    add_min_uptime_constraints(model, u, v, generators, timesteps)
    add_min_downtime_constraints(model, u, w, generators, timesteps)
    add_ramp_constraints(model, p, u, generators, timesteps)
    add_maintenance_constraints(model, u, generators, timesteps)
    add_reserve_margin_constraints(
        model, u, generators, timesteps, demand, reserve_margin
    )
    add_storage_soc_constraints(
        model, p, p_ch or {}, p_dis or {}, z_ch or {}, soc or {},
        u, generators, timesteps, period_duration_h,
    )


def _split_demand_by_region(
    generators: List[Generator],
    demand: List[float],
) -> Dict[str, List[float]]:
    """Split total demand proportionally by region capacity.

    Each region receives a fraction of total demand proportional to its
    share of total generation capacity.  Follows the same pattern as
    ``_split_demand_by_capacity`` in ``decomposition.py``.

    Args:
        generators: List of generators with ``region`` and
            ``capacity_mw`` attributes.
        demand: System-wide demand values (MW) per period.

    Returns:
        Mapping of region identifier to demand series for that region.
    """
    # Group generators by region
    groups: Dict[str, List[Generator]] = {}
    for g in generators:
        key = g.region if g.region else "_unassigned"
        groups.setdefault(key, []).append(g)

    total_cap = sum(g.capacity_mw for g in generators)
    if total_cap <= 0:
        # Equal split if no capacity info
        n = len(groups) or 1
        return {key: [d / n for d in demand] for key in groups}

    result: Dict[str, List[float]] = {}
    for key, gens in groups.items():
        group_cap = sum(g.capacity_mw for g in gens)
        fraction = group_cap / total_cap
        result[key] = [d * fraction for d in demand]

    return result


def _select_solver(params: UCParameters) -> pulp.apis.LpSolver:
    """Select and configure the MIP solver backend.

    Prefers HiGHS_CMD when available, falling back to PULP_CBC_CMD.
    Applies time limit and MIP gap settings from UCParameters.

    Args:
        params: UC parameters with solver configuration.

    Returns:
        Configured PuLP solver instance.
    """
    solver_kwargs: Dict = {"msg": 0}

    if params.solver_time_limit_s is not None:
        solver_kwargs["timeLimit"] = params.solver_time_limit_s

    if params.mip_gap is not None:
        solver_kwargs["gapRel"] = params.mip_gap

    # Try HiGHS first if requested
    if params.solver_name.upper() in ("HIGHS", "HIGHS_CMD"):
        try:
            solver = pulp.HiGHS_CMD(**solver_kwargs)
            if solver.available():
                logger.info("Using HiGHS solver")
                return solver
        except Exception:
            logger.info("HiGHS solver not available, trying CBC fallback")

    # Fallback to CBC (bundled with PuLP)
    try:
        solver = pulp.PULP_CBC_CMD(**solver_kwargs)
        if solver.available():
            logger.info("Using CBC solver (PuLP bundled)")
            return solver
    except Exception:
        pass

    # Last resort: default solver
    logger.warning("No preferred solver available; using PuLP default")
    return pulp.PULP_CBC_CMD(msg=0)


def _extract_solution(
    result: UCResult,
    model: pulp.LpProblem,
    u: Dict[Tuple[str, int], pulp.LpVariable],
    p: Dict[Tuple[str, int], pulp.LpVariable],
    v: Dict[Tuple[str, int], pulp.LpVariable],
    w: Dict[Tuple[str, int], pulp.LpVariable],
    generators: List[Generator],
    timesteps: List[int],
    *,
    p_ch: Optional[Dict[Tuple[str, int], pulp.LpVariable]] = None,
    p_dis: Optional[Dict[Tuple[str, int], pulp.LpVariable]] = None,
    soc: Optional[Dict[Tuple[str, int], pulp.LpVariable]] = None,
    storage_ids: Optional[set] = None,
    interconnections: Optional[List[Interconnection]] = None,
    f: Optional[Dict[Tuple[str, int], pulp.LpVariable]] = None,
) -> None:
    """Extract solution values into GeneratorSchedule objects.

    Populates the UCResult with per-generator schedules, cost
    breakdowns, and the total system cost.  For storage generators,
    also extracts SOC, charge, and discharge profiles.  For
    interconnections, extracts per-interconnection flow values and
    logs saturation summary.
    """
    result.total_cost = pulp.value(model.objective)

    if p_ch is None:
        p_ch = {}
    if p_dis is None:
        p_dis = {}
    if soc is None:
        soc = {}
    if storage_ids is None:
        storage_ids = set()

    # Try to extract MIP gap from the solver
    try:
        if hasattr(model, "sol_status") and model.sol_status == 1:
            result.gap = 0.0
    except Exception:
        pass

    for g in generators:
        commitment = []
        power_output = []
        gen_fuel_cost = 0.0
        gen_noload_cost = 0.0
        gen_startup_cost = 0.0
        gen_shutdown_cost = 0.0

        # Storage-specific result arrays
        soc_mwh = []
        charge_mw = []
        discharge_mw = []
        is_storage = g.id in storage_ids

        for t in timesteps:
            u_val = int(round(pulp.value(u[(g.id, t)])))
            p_val = float(pulp.value(p[(g.id, t)]))
            v_val = int(round(pulp.value(v[(g.id, t)])))
            w_val = int(round(pulp.value(w[(g.id, t)])))

            commitment.append(u_val)
            power_output.append(round(p_val, 6))

            gen_fuel_cost += g.fuel_cost_per_mwh * p_val
            gen_noload_cost += (g.no_load_cost + g.labor_cost_per_h) * u_val
            gen_startup_cost += g.startup_cost * v_val
            gen_shutdown_cost += g.shutdown_cost * w_val

            if is_storage:
                soc_mwh.append(round(float(pulp.value(soc[(g.id, t)])), 6))
                charge_mw.append(round(float(pulp.value(p_ch[(g.id, t)])), 6))
                discharge_mw.append(round(float(pulp.value(p_dis[(g.id, t)])), 6))

        schedule = GeneratorSchedule(
            generator_id=g.id,
            commitment=commitment,
            power_output_mw=power_output,
            startup_cost=gen_startup_cost,
            shutdown_cost=gen_shutdown_cost,
            fuel_cost=gen_fuel_cost,
            no_load_cost=gen_noload_cost,
            soc_mwh=soc_mwh,
            charge_mw=charge_mw,
            discharge_mw=discharge_mw,
        )
        result.schedules.append(schedule)

        logger.info(
            "Generator %s: committed=%d/%d periods, cost=%.1f",
            g.id,
            sum(commitment),
            len(timesteps),
            schedule.total_cost,
        )

    # --- Extract interconnection flow results ------------------------------
    if interconnections and f:
        saturated_ids = []
        for ic in interconnections:
            flow_values = []
            for t in timesteps:
                flow_val = float(pulp.value(f[(ic.id, t)]))
                flow_values.append(round(flow_val, 6))

            ic_flow = InterconnectionFlow(
                interconnection_id=ic.id,
                flow_mw=flow_values,
            )
            result.interconnection_flows.append(ic_flow)

            max_abs_flow = max(abs(fv) for fv in flow_values) if flow_values else 0.0
            is_saturated = max_abs_flow >= ic.capacity_mw * 0.999
            if is_saturated:
                saturated_ids.append(ic.id)

            logger.info(
                "Interconnection %s: max_flow=%.1f MW / %.1f MW capacity%s",
                ic.id,
                max_abs_flow,
                ic.capacity_mw,
                " (SATURATED)" if is_saturated else "",
            )

        if saturated_ids:
            logger.info(
                "Saturated interconnections: %s",
                ", ".join(saturated_ids),
            )


def _diagnose_infeasibility(
    result: UCResult,
    generators: List[Generator],
    timesteps: List[int],
    demand: List[float],
) -> None:
    """Add diagnostic information when the solver reports infeasibility.

    Identifies timesteps where demand exceeds total available capacity
    (accounting for maintenance) and reports the shortfall.
    """
    infeasible_periods = []
    for idx, t in enumerate(timesteps):
        available_cap = 0.0
        maint_count = 0
        for g in generators:
            in_maintenance = any(
                start_h <= t < end_h for start_h, end_h in g.maintenance_windows
            )
            if in_maintenance:
                maint_count += 1
            else:
                available_cap += g.capacity_mw
        if available_cap < demand[idx]:
            infeasible_periods.append(
                (t, demand[idx], available_cap, maint_count)
            )

    if infeasible_periods:
        for t, d, cap, mc in infeasible_periods:
            msg = (
                f"Infeasible at t={t}: demand={d:.1f} MW, "
                f"available_capacity={cap:.1f} MW, "
                f"shortfall={d - cap:.1f} MW"
            )
            if mc > 0:
                msg += f", generators_in_maintenance={mc}"
            result.warnings.append(msg)
            logger.warning(msg)
    else:
        msg = (
            "Infeasibility detected but capacity appears sufficient at "
            "all timesteps. Check min-uptime, min-downtime, or ramp "
            "constraints for conflicting requirements."
        )
        result.warnings.append(msg)
        logger.warning(msg)
