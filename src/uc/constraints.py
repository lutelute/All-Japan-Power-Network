"""Modular constraint builders for the unit commitment MILP formulation.

Each function adds a specific class of constraints to a PuLP model,
keeping the solver code decomposed and testable.  Constraint names
follow the pattern ``<type>_<generator>_t<period>`` so that
infeasibility diagnostics can pinpoint which constraint is violated.

Usage::

    import pulp
    from src.uc.constraints import (
        add_demand_balance_constraints,
        add_capacity_bounds_constraints,
        add_startup_shutdown_logic,
        add_min_uptime_constraints,
        add_min_downtime_constraints,
        add_ramp_constraints,
        add_maintenance_constraints,
        add_reserve_margin_constraints,
        add_storage_soc_constraints,
        add_transmission_capacity_constraints,
        add_nodal_balance_constraints,
    )

    model = pulp.LpProblem("UC", pulp.LpMinimize)
    # ... create variables u, p, v, w ...
    add_demand_balance_constraints(model, p, generators, timesteps, demand)
    add_capacity_bounds_constraints(model, u, p, generators, timesteps)
    # ... etc ...
"""

from typing import Dict, List, Tuple

import pulp

from src.model.generator import Generator
from src.uc.models import Interconnection
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def add_demand_balance_constraints(
    model: pulp.LpProblem,
    p: Dict[Tuple[str, int], pulp.LpVariable],
    generators: List[Generator],
    timesteps: List[int],
    demand: List[float],
) -> int:
    """Add system-wide demand balance constraints.

    Ensures that total generation meets or exceeds demand at every
    time step:

        ``Σ_g p[g,t] >= demand[t]``  for all *t*

    Args:
        model: PuLP model to add constraints to.
        p: Power output variables indexed by ``(generator_id, timestep)``.
        generators: List of generators available for commitment.
        timesteps: List of time period indices.
        demand: Demand values (MW) indexed to match *timesteps*.

    Returns:
        Number of constraints added.
    """
    gen_ids = [g.id for g in generators]
    count = 0
    for idx, t in enumerate(timesteps):
        model += (
            pulp.lpSum(p[(g_id, t)] for g_id in gen_ids) >= demand[idx],
            f"demand_balance_t{t}",
        )
        count += 1
    logger.info(
        "Added %d demand balance constraints (%d timesteps)",
        count,
        len(timesteps),
    )
    return count


def add_capacity_bounds_constraints(
    model: pulp.LpProblem,
    u: Dict[Tuple[str, int], pulp.LpVariable],
    p: Dict[Tuple[str, int], pulp.LpVariable],
    generators: List[Generator],
    timesteps: List[int],
) -> int:
    """Add per-generator capacity bound constraints.

    When a generator is committed (``u=1``), its output must lie between
    its minimum and maximum capacity.  When off (``u=0``), output is
    forced to zero:

        ``p_min[g] * u[g,t] <= p[g,t] <= p_max[g] * u[g,t]``

    Args:
        model: PuLP model to add constraints to.
        u: Binary commitment variables indexed by ``(generator_id, timestep)``.
        p: Power output variables indexed by ``(generator_id, timestep)``.
        generators: List of generators available for commitment.
        timesteps: List of time period indices.

    Returns:
        Number of constraints added.
    """
    count = 0
    n_skipped = 0
    for g in generators:
        # Storage generators are bounded by charge/discharge limit
        # constraints instead of standard capacity bounds (which assume p >= 0).
        if hasattr(g, "is_storage") and g.is_storage:
            n_skipped += 1
            continue
        for t in timesteps:
            # Lower bound: p >= p_min * u
            model += (
                p[(g.id, t)] >= g.p_min_mw * u[(g.id, t)],
                f"cap_lb_{g.id}_t{t}",
            )
            count += 1
            # Upper bound: p <= p_max * u
            model += (
                p[(g.id, t)] <= g.capacity_mw * u[(g.id, t)],
                f"cap_ub_{g.id}_t{t}",
            )
            count += 1
    logger.info(
        "Added %d capacity bound constraints (%d generators × %d timesteps × 2, %d storage skipped)",
        count,
        len(generators) - n_skipped,
        len(timesteps),
        n_skipped,
    )
    return count


def add_startup_shutdown_logic(
    model: pulp.LpProblem,
    u: Dict[Tuple[str, int], pulp.LpVariable],
    v: Dict[Tuple[str, int], pulp.LpVariable],
    w: Dict[Tuple[str, int], pulp.LpVariable],
    generators: List[Generator],
    timesteps: List[int],
) -> int:
    """Add startup/shutdown indicator linking constraints.

    Links the binary startup (*v*) and shutdown (*w*) indicators to
    commitment status changes:

        ``v[g,t] - w[g,t] = u[g,t] - u[g,t-1]``  for all *g*, *t*

    For the first time step, ``u[g,t-1]`` is assumed to be 0 (all
    generators start offline).

    Args:
        model: PuLP model to add constraints to.
        u: Binary commitment variables indexed by ``(generator_id, timestep)``.
        v: Binary startup indicator variables indexed by
            ``(generator_id, timestep)``.
        w: Binary shutdown indicator variables indexed by
            ``(generator_id, timestep)``.
        generators: List of generators available for commitment.
        timesteps: List of time period indices (must be sorted).

    Returns:
        Number of constraints added.
    """
    count = 0
    for g in generators:
        for i, t in enumerate(timesteps):
            if i == 0:
                # First period: assume generator was off (u[g, t-1] = 0)
                model += (
                    v[(g.id, t)] - w[(g.id, t)] == u[(g.id, t)],
                    f"startup_shutdown_{g.id}_t{t}",
                )
            else:
                t_prev = timesteps[i - 1]
                model += (
                    v[(g.id, t)] - w[(g.id, t)]
                    == u[(g.id, t)] - u[(g.id, t_prev)],
                    f"startup_shutdown_{g.id}_t{t}",
                )
            count += 1
    logger.info(
        "Added %d startup/shutdown logic constraints (%d generators × %d timesteps)",
        count,
        len(generators),
        len(timesteps),
    )
    return count


def add_min_uptime_constraints(
    model: pulp.LpProblem,
    u: Dict[Tuple[str, int], pulp.LpVariable],
    v: Dict[Tuple[str, int], pulp.LpVariable],
    generators: List[Generator],
    timesteps: List[int],
) -> int:
    """Add minimum up-time constraints.

    Once started, a generator must remain on for at least
    ``min_up_time_h`` consecutive periods:

        ``Σ_{τ=t-MUT+1}^{t} v[g,τ] <= u[g,t]``  for all *g*, *t*

    This ensures that if any startup occurred in the look-back window
    of length *MUT*, the generator must still be committed at *t*.

    Generators with ``min_up_time_h <= 1`` are skipped (no constraint
    needed beyond the single-period startup).

    Args:
        model: PuLP model to add constraints to.
        u: Binary commitment variables indexed by ``(generator_id, timestep)``.
        v: Binary startup indicator variables indexed by
            ``(generator_id, timestep)``.
        generators: List of generators available for commitment.
        timesteps: List of time period indices (must be sorted).

    Returns:
        Number of constraints added.
    """
    count = 0
    for g in generators:
        mut = g.min_up_time_h
        if mut <= 1:
            continue
        for i, t in enumerate(timesteps):
            # Look-back window: indices [max(0, i-mut+1), i]
            start_idx = max(0, i - mut + 1)
            window = timesteps[start_idx : i + 1]
            model += (
                pulp.lpSum(v[(g.id, tau)] for tau in window) <= u[(g.id, t)],
                f"min_up_{g.id}_t{t}",
            )
            count += 1
    n_skipped = sum(1 for g in generators if g.min_up_time_h <= 1)
    logger.info(
        "Added %d min-uptime constraints (%d generators active, %d skipped with MUT<=1)",
        count,
        len(generators) - n_skipped,
        n_skipped,
    )
    return count


def add_min_downtime_constraints(
    model: pulp.LpProblem,
    u: Dict[Tuple[str, int], pulp.LpVariable],
    w: Dict[Tuple[str, int], pulp.LpVariable],
    generators: List[Generator],
    timesteps: List[int],
) -> int:
    """Add minimum down-time constraints.

    Once shut down, a generator must remain off for at least
    ``min_down_time_h`` consecutive periods:

        ``Σ_{τ=t-MDT+1}^{t} w[g,τ] <= 1 - u[g,t]``  for all *g*, *t*

    This ensures that if any shutdown occurred in the look-back window
    of length *MDT*, the generator must still be offline at *t*.

    Generators with ``min_down_time_h <= 1`` are skipped (no constraint
    needed beyond the single-period shutdown).

    Args:
        model: PuLP model to add constraints to.
        u: Binary commitment variables indexed by ``(generator_id, timestep)``.
        w: Binary shutdown indicator variables indexed by
            ``(generator_id, timestep)``.
        generators: List of generators available for commitment.
        timesteps: List of time period indices (must be sorted).

    Returns:
        Number of constraints added.
    """
    count = 0
    for g in generators:
        mdt = g.min_down_time_h
        if mdt <= 1:
            continue
        for i, t in enumerate(timesteps):
            # Look-back window: indices [max(0, i-mdt+1), i]
            start_idx = max(0, i - mdt + 1)
            window = timesteps[start_idx : i + 1]
            model += (
                pulp.lpSum(w[(g.id, tau)] for tau in window) <= 1 - u[(g.id, t)],
                f"min_down_{g.id}_t{t}",
            )
            count += 1
    n_skipped = sum(1 for g in generators if g.min_down_time_h <= 1)
    logger.info(
        "Added %d min-downtime constraints (%d generators active, %d skipped with MDT<=1)",
        count,
        len(generators) - n_skipped,
        n_skipped,
    )
    return count


def add_ramp_constraints(
    model: pulp.LpProblem,
    p: Dict[Tuple[str, int], pulp.LpVariable],
    u: Dict[Tuple[str, int], pulp.LpVariable],
    generators: List[Generator],
    timesteps: List[int],
) -> int:
    """Add ramp-rate limit constraints.

    Limits the change in power output between consecutive periods.
    Uses a big-M relaxation so that ramp constraints do not bind when
    a generator is starting up or shutting down:

        Ramp up:   ``p[g,t] - p[g,t-1] <= ramp_up[g] + Pmax[g] * (1 - u[g,t-1])``
        Ramp down: ``p[g,t-1] - p[g,t] <= ramp_down[g] + Pmax[g] * (1 - u[g,t])``

    When the generator was on in the previous period (``u[g,t-1]=1``),
    the big-M term vanishes and the ramp rate binds.  During startup or
    shutdown the constraint is relaxed.

    Generators with ``ramp_up_mw_per_h is None`` (unlimited ramp-up) or
    ``ramp_down_mw_per_h is None`` (unlimited ramp-down) skip the
    corresponding constraint.

    Args:
        model: PuLP model to add constraints to.
        p: Power output variables indexed by ``(generator_id, timestep)``.
        u: Binary commitment variables indexed by ``(generator_id, timestep)``.
        generators: List of generators available for commitment.
        timesteps: List of time period indices (must be sorted).

    Returns:
        Number of constraints added.
    """
    count = 0
    for g in generators:
        has_ramp_up = g.ramp_up_mw_per_h is not None
        has_ramp_down = g.ramp_down_mw_per_h is not None
        if not has_ramp_up and not has_ramp_down:
            continue
        for i in range(1, len(timesteps)):
            t = timesteps[i]
            t_prev = timesteps[i - 1]
            # Ramp-up constraint
            if has_ramp_up:
                model += (
                    p[(g.id, t)] - p[(g.id, t_prev)]
                    <= g.ramp_up_mw_per_h + g.capacity_mw * (1 - u[(g.id, t_prev)]),
                    f"ramp_up_{g.id}_t{t}",
                )
                count += 1
            # Ramp-down constraint
            if has_ramp_down:
                model += (
                    p[(g.id, t_prev)] - p[(g.id, t)]
                    <= g.ramp_down_mw_per_h + g.capacity_mw * (1 - u[(g.id, t)]),
                    f"ramp_down_{g.id}_t{t}",
                )
                count += 1
    n_unlimited = sum(
        1
        for g in generators
        if g.ramp_up_mw_per_h is None and g.ramp_down_mw_per_h is None
    )
    logger.info(
        "Added %d ramp constraints (%d generators with limits, %d unlimited)",
        count,
        len(generators) - n_unlimited,
        n_unlimited,
    )
    return count


def add_maintenance_constraints(
    model: pulp.LpProblem,
    u: Dict[Tuple[str, int], pulp.LpVariable],
    generators: List[Generator],
    timesteps: List[int],
) -> int:
    """Add maintenance window constraints.

    Forces a generator offline during its scheduled maintenance
    windows:

        ``u[g,t] = 0``  for all *g*, *t* where *t* falls within a
        maintenance window of generator *g*.

    A time step *t* falls within window ``(start_h, end_h)`` when
    ``start_h <= t < end_h``.

    Args:
        model: PuLP model to add constraints to.
        u: Binary commitment variables indexed by ``(generator_id, timestep)``.
        generators: List of generators available for commitment.
        timesteps: List of time period indices.

    Returns:
        Number of constraints added.
    """
    count = 0
    for g in generators:
        if not g.maintenance_windows:
            continue
        # Build a set of timesteps under maintenance for fast lookup
        maint_periods = set()
        for start_h, end_h in g.maintenance_windows:
            for t in timesteps:
                if start_h <= t < end_h:
                    maint_periods.add(t)
        for t in sorted(maint_periods):
            model += (
                u[(g.id, t)] == 0,
                f"maint_{g.id}_t{t}",
            )
            count += 1
        if maint_periods:
            logger.info(
                "Generator %s: forced offline for %d maintenance periods",
                g.id,
                len(maint_periods),
            )
    logger.info("Added %d maintenance constraints in total", count)
    return count


def add_reserve_margin_constraints(
    model: pulp.LpProblem,
    u: Dict[Tuple[str, int], pulp.LpVariable],
    generators: List[Generator],
    timesteps: List[int],
    demand: List[float],
    reserve_margin: float,
) -> int:
    """Add spinning-reserve margin constraints.

    Ensures that committed capacity headroom exceeds demand by the
    required reserve margin at every time step:

        ``Σ_g Pmax[g] * u[g,t] >= demand[t] * (1 + reserve_margin)``

    This guarantees that committed generators have enough total
    capacity to absorb demand spikes without requiring additional
    startups.

    When *reserve_margin* is 0.0 (or negative), no constraints are
    added.

    Args:
        model: PuLP model to add constraints to.
        u: Binary commitment variables indexed by ``(generator_id, timestep)``.
        generators: List of generators available for commitment.
        timesteps: List of time period indices.
        demand: Demand values (MW) indexed to match *timesteps*.
        reserve_margin: Required reserve as a fraction of demand
            (e.g., 0.10 for 10%).

    Returns:
        Number of constraints added.
    """
    if reserve_margin <= 0:
        logger.info("Reserve margin is %.2f; skipping reserve constraints", reserve_margin)
        return 0

    count = 0
    for idx, t in enumerate(timesteps):
        required = demand[idx] * (1.0 + reserve_margin)
        model += (
            pulp.lpSum(g.capacity_mw * u[(g.id, t)] for g in generators)
            >= required,
            f"reserve_margin_t{t}",
        )
        count += 1
    logger.info(
        "Added %d reserve margin constraints (margin=%.1f%%)",
        count,
        reserve_margin * 100,
    )
    return count


def add_storage_soc_constraints(
    model: pulp.LpProblem,
    p: Dict[Tuple[str, int], pulp.LpVariable],
    p_ch: Dict[Tuple[str, int], pulp.LpVariable],
    p_dis: Dict[Tuple[str, int], pulp.LpVariable],
    z_ch: Dict[Tuple[str, int], pulp.LpVariable],
    soc: Dict[Tuple[str, int], pulp.LpVariable],
    u: Dict[Tuple[str, int], pulp.LpVariable],
    generators: List[Generator],
    timesteps: List[int],
    period_duration_h: float = 1.0,
) -> int:
    """Add state-of-charge (SOC) constraints for storage generators.

    Implements time-coupled energy storage constraints for both pumped
    hydro and battery generators.  Non-storage generators (where
    ``g.is_storage`` is False) are skipped.

    For each storage generator *g* at each timestep *t*, the following
    constraints are added:

    1. **SOC balance** (time-coupled):
        ``soc[g,t] = soc[g,t-1] + η_ch × p_ch[g,t] × Δt
        - (1/η_dis) × p_dis[g,t] × Δt``
        For t=0, ``soc[g,t-1]`` is the initial SOC
        (``initial_soc_fraction × storage_capacity_mwh``).

    2. **SOC bounds**: Enforced via variable bounds
        (``lowBound=0, upBound=storage_capacity_mwh``).

    3. **Charge power limit**:
        ``p_ch[g,t] <= charge_rate_mw × z_ch[g,t]``

    4. **Discharge power limit**:
        ``p_dis[g,t] <= discharge_rate_mw × (1 - z_ch[g,t])``

    5. **Net power linking**:
        ``p[g,t] = p_dis[g,t] - p_ch[g,t]``

    6. **Commitment linking**:
        ``p_ch[g,t] <= charge_rate_mw × u[g,t]``
        ``p_dis[g,t] <= discharge_rate_mw × u[g,t]``

    7. **Terminal SOC** (one per storage generator):
        ``soc[g,T] >= min_terminal_soc_fraction × storage_capacity_mwh``

    Args:
        model: PuLP model to add constraints to.
        p: Net power output variables indexed by
            ``(generator_id, timestep)``.
        p_ch: Charging power variables indexed by
            ``(generator_id, timestep)``.
        p_dis: Discharging power variables indexed by
            ``(generator_id, timestep)``.
        z_ch: Binary charge-mode indicator variables indexed by
            ``(generator_id, timestep)``.
        soc: State-of-charge variables indexed by
            ``(generator_id, timestep)``.
        u: Binary commitment variables indexed by
            ``(generator_id, timestep)``.
        generators: List of generators (storage and non-storage).
        timesteps: List of time period indices (must be sorted).
        period_duration_h: Duration of each period in hours.

    Returns:
        Number of constraints added.
    """
    count = 0
    n_storage = 0
    n_skipped = 0

    for g in generators:
        if not (hasattr(g, "is_storage") and g.is_storage):
            n_skipped += 1
            continue

        n_storage += 1

        # Resolve charge/discharge rates (None defaults to capacity_mw)
        charge_rate = g.charge_rate_mw if g.charge_rate_mw is not None else g.capacity_mw
        discharge_rate = (
            g.discharge_rate_mw if g.discharge_rate_mw is not None else g.capacity_mw
        )
        eta_ch = g.charge_efficiency
        eta_dis = g.discharge_efficiency
        capacity_mwh = g.storage_capacity_mwh
        initial_soc = g.initial_soc_fraction * capacity_mwh
        dt = period_duration_h

        for i, t in enumerate(timesteps):
            # (1) SOC balance
            if i == 0:
                # First period: use initial SOC as soc[g, t-1]
                model += (
                    soc[(g.id, t)]
                    == initial_soc
                    + eta_ch * p_ch[(g.id, t)] * dt
                    - (1.0 / eta_dis) * p_dis[(g.id, t)] * dt,
                    f"soc_balance_{g.id}_t{t}",
                )
            else:
                t_prev = timesteps[i - 1]
                model += (
                    soc[(g.id, t)]
                    == soc[(g.id, t_prev)]
                    + eta_ch * p_ch[(g.id, t)] * dt
                    - (1.0 / eta_dis) * p_dis[(g.id, t)] * dt,
                    f"soc_balance_{g.id}_t{t}",
                )
            count += 1

            # (2) SOC bounds — enforced via variable bounds (set externally)

            # (3) Charge power limit: p_ch <= charge_rate * z_ch
            model += (
                p_ch[(g.id, t)] <= charge_rate * z_ch[(g.id, t)],
                f"charge_limit_{g.id}_t{t}",
            )
            count += 1

            # (4) Discharge power limit: p_dis <= discharge_rate * (1 - z_ch)
            model += (
                p_dis[(g.id, t)] <= discharge_rate * (1 - z_ch[(g.id, t)]),
                f"discharge_limit_{g.id}_t{t}",
            )
            count += 1

            # (5) Net power linking: p = p_dis - p_ch
            model += (
                p[(g.id, t)] == p_dis[(g.id, t)] - p_ch[(g.id, t)],
                f"net_power_{g.id}_t{t}",
            )
            count += 1

            # (6) Commitment linking: charge and discharge require commitment
            model += (
                p_ch[(g.id, t)] <= charge_rate * u[(g.id, t)],
                f"commit_ch_{g.id}_t{t}",
            )
            count += 1
            model += (
                p_dis[(g.id, t)] <= discharge_rate * u[(g.id, t)],
                f"commit_dis_{g.id}_t{t}",
            )
            count += 1

        # (7) Terminal SOC: soc[g,T] >= min_terminal_soc_fraction * capacity
        if timesteps and g.min_terminal_soc_fraction > 0:
            t_last = timesteps[-1]
            model += (
                soc[(g.id, t_last)]
                >= g.min_terminal_soc_fraction * capacity_mwh,
                f"terminal_soc_{g.id}",
            )
            count += 1

    logger.info(
        "Added %d storage SOC constraints (%d storage generators, %d non-storage skipped)",
        count,
        n_storage,
        n_skipped,
    )
    return count


def add_transmission_capacity_constraints(
    model: pulp.LpProblem,
    f: Dict[Tuple[str, int], pulp.LpVariable],
    interconnections: List[Interconnection],
    timesteps: List[int],
) -> int:
    """Add transmission capacity constraints for interconnections.

    Limits the power flow on each interconnection to its rated capacity
    in both directions:

        ``f[ic,t] <= capacity_mw[ic]``   (upper bound)
        ``f[ic,t] >= -capacity_mw[ic]``  (lower bound)

    Positive flow represents power transfer from ``from_region`` to
    ``to_region``; negative flow represents the reverse direction.

    Args:
        model: PuLP model to add constraints to.
        f: Flow variables indexed by ``(interconnection_id, timestep)``.
        interconnections: List of interconnections to constrain.
        timesteps: List of time period indices.

    Returns:
        Number of constraints added.
    """
    count = 0
    for ic in interconnections:
        for t in timesteps:
            # Upper bound: f <= capacity
            model += (
                f[(ic.id, t)] <= ic.capacity_mw,
                f"tx_cap_ub_{ic.id}_t{t}",
            )
            count += 1
            # Lower bound: f >= -capacity
            model += (
                f[(ic.id, t)] >= -ic.capacity_mw,
                f"tx_cap_lb_{ic.id}_t{t}",
            )
            count += 1
    logger.info(
        "Added %d transmission capacity constraints (%d interconnections × %d timesteps × 2)",
        count,
        len(interconnections),
        len(timesteps),
    )
    return count


def add_nodal_balance_constraints(
    model: pulp.LpProblem,
    p: Dict[Tuple[str, int], pulp.LpVariable],
    f: Dict[Tuple[str, int], pulp.LpVariable],
    generators: List[Generator],
    interconnections: List[Interconnection],
    timesteps: List[int],
    regional_demand: Dict[str, List[float]],
) -> int:
    """Add nodal (per-region) power balance constraints.

    Ensures that generation plus net imports meets or exceeds demand in
    each region at every time step:

        ``Σ_{g∈r} p[g,t] + Σ_{ic: to=r} f[ic,t]
        - Σ_{ic: from=r} f[ic,t] >= demand[r][idx]``  for all *r*, *t*

    Positive flow on an interconnection represents power transfer from
    ``from_region`` to ``to_region``.

    Args:
        model: PuLP model to add constraints to.
        p: Power output variables indexed by ``(generator_id, timestep)``.
        f: Flow variables indexed by ``(interconnection_id, timestep)``.
        generators: List of generators available for commitment.
        interconnections: List of inter-regional interconnections.
        timesteps: List of time period indices.
        regional_demand: Mapping of region identifier to demand series
            (MW) aligned with *timesteps*.

    Returns:
        Number of constraints added.
    """
    # Group generator IDs by region
    region_gen_ids: Dict[str, List[str]] = {}
    for g in generators:
        region_gen_ids.setdefault(g.region, []).append(g.id)

    # Group interconnections by to_region and from_region
    ic_to_region: Dict[str, List[Interconnection]] = {}
    ic_from_region: Dict[str, List[Interconnection]] = {}
    for ic in interconnections:
        ic_to_region.setdefault(ic.to_region, []).append(ic)
        ic_from_region.setdefault(ic.from_region, []).append(ic)

    count = 0
    for region, demands in regional_demand.items():
        gen_ids = region_gen_ids.get(region, [])
        inflows = ic_to_region.get(region, [])
        outflows = ic_from_region.get(region, [])

        for idx, t in enumerate(timesteps):
            generation = pulp.lpSum(p[(g_id, t)] for g_id in gen_ids)
            imports = pulp.lpSum(f[(ic.id, t)] for ic in inflows)
            exports = pulp.lpSum(f[(ic.id, t)] for ic in outflows)

            model += (
                generation + imports - exports >= demands[idx],
                f"nodal_bal_{region}_t{t}",
            )
            count += 1

    logger.info(
        "Added %d nodal balance constraints (%d regions × %d timesteps)",
        count,
        len(regional_demand),
        len(timesteps),
    )
    return count
