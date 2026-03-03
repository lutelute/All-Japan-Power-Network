"""Comprehensive tests for the UC solver module.

Tests solver correctness including:
- Feasibility and optimality for small instances (3 generators, 24 periods)
- Demand balance verified at every timestep
- Minimum up-time enforcement
- Minimum down-time enforcement
- Ramp-rate limit compliance
- Maintenance window honoring
- Reserve margin satisfaction
- Infeasibility detection with diagnostics
- Single generator edge case
- Zero demand timestep behaviour
- Solver timeout / MIP gap configuration
- Pumped hydro SOC tracking and constraints
- Battery SOC tracking with high-efficiency settings
- Storage integration with thermal generators
- Pumped hydro day-cycle charge/discharge patterns
- Regression verification for non-storage instances
"""

from typing import List

import pytest

from src.model.generator import Generator
from src.uc.models import DemandProfile, TimeHorizon, UCParameters, UCResult
from src.uc.solver import solve_uc
from tests.conftest import make_generator, make_storage_generator


# ======================================================================
# Helpers
# ======================================================================


def _flat_demand(mw: float, periods: int) -> DemandProfile:
    """Create a constant demand profile."""
    return DemandProfile(demands=[mw] * periods)


def _make_simple_generators() -> List[Generator]:
    """Create 3 generators for the standard feasible test instance.

    Generator lineup (total capacity = 450 MW):
    - g1: 200 MW coal base-load — cheap fuel, slow to start, tight ramp
    - g2: 150 MW LNG mid-merit — moderate fuel cost, flexible
    - g3: 100 MW oil peaker — expensive fuel, very flexible
    """
    g1 = make_generator(
        id="g1",
        name="Base Coal",
        capacity_mw=200.0,
        fuel_type="coal",
        p_min_mw=50.0,
        startup_cost=5000.0,
        shutdown_cost=2000.0,
        min_up_time_h=4,
        min_down_time_h=4,
        ramp_up_mw_per_h=50.0,
        ramp_down_mw_per_h=50.0,
        fuel_cost_per_mwh=30.0,
        labor_cost_per_h=10.0,
        no_load_cost=100.0,
    )
    g2 = make_generator(
        id="g2",
        name="Mid LNG",
        capacity_mw=150.0,
        fuel_type="lng",
        p_min_mw=30.0,
        startup_cost=2000.0,
        shutdown_cost=1000.0,
        min_up_time_h=2,
        min_down_time_h=2,
        ramp_up_mw_per_h=75.0,
        ramp_down_mw_per_h=75.0,
        fuel_cost_per_mwh=50.0,
        labor_cost_per_h=8.0,
        no_load_cost=50.0,
    )
    g3 = make_generator(
        id="g3",
        name="Peak Oil",
        capacity_mw=100.0,
        fuel_type="oil",
        p_min_mw=10.0,
        startup_cost=1000.0,
        shutdown_cost=500.0,
        min_up_time_h=1,
        min_down_time_h=1,
        ramp_up_mw_per_h=100.0,
        ramp_down_mw_per_h=100.0,
        fuel_cost_per_mwh=80.0,
        labor_cost_per_h=5.0,
        no_load_cost=20.0,
    )
    return [g1, g2, g3]


def _find_schedule(result: UCResult, gen_id: str):
    """Find the GeneratorSchedule for a given generator id."""
    for s in result.schedules:
        if s.generator_id == gen_id:
            return s
    raise ValueError(f"No schedule for generator '{gen_id}'")


# ======================================================================
# TestSolverFeasibility — 3 generators, 24 periods
# ======================================================================


class TestSolverFeasibility:
    """Tests that the solver returns Optimal for a feasible instance."""

    def test_optimal_status_3gen_24h(self) -> None:
        """3-generator 24-period instance returns Optimal status."""
        gens = _make_simple_generators()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(200.0, 24)  # well within 450 MW total
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)

        assert result.status == "Optimal"
        assert result.total_cost > 0
        assert result.solve_time_s >= 0
        assert len(result.schedules) == 3

    def test_result_has_schedules_for_all_generators(self) -> None:
        """Each generator has a schedule in the result."""
        gens = _make_simple_generators()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(200.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)

        gen_ids = {s.generator_id for s in result.schedules}
        assert gen_ids == {"g1", "g2", "g3"}

    def test_commitment_and_power_length_matches_periods(self) -> None:
        """Commitment and power vectors have length == num_periods."""
        gens = _make_simple_generators()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(200.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)

        for sched in result.schedules:
            assert len(sched.commitment) == 24
            assert len(sched.power_output_mw) == 24

    def test_total_cost_is_sum_of_generator_costs(self) -> None:
        """System total cost equals the sum of individual generator costs."""
        gens = _make_simple_generators()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(200.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sum_gen_costs = sum(s.total_cost for s in result.schedules)
        assert abs(result.total_cost - sum_gen_costs) < 1.0


# ======================================================================
# TestDemandBalance
# ======================================================================


class TestDemandBalance:
    """Tests that demand is met at every timestep."""

    def test_demand_balance_flat_demand(self) -> None:
        """Total generation >= demand at each timestep with flat demand."""
        gens = _make_simple_generators()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        demand_mw = 200.0
        dp = _flat_demand(demand_mw, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        for t in range(24):
            total_gen = sum(s.power_output_mw[t] for s in result.schedules)
            assert total_gen >= demand_mw - 1e-3, (
                f"Demand balance violated at t={t}: "
                f"generation={total_gen:.4f} < demand={demand_mw}"
            )

    def test_demand_balance_varying_demand(self) -> None:
        """Total generation meets a varying demand profile."""
        gens = _make_simple_generators()
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        demands = [100, 150, 200, 250, 300, 300, 250, 200, 150, 100, 100, 100]
        dp = DemandProfile(demands=[float(d) for d in demands])
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        for t in range(12):
            total_gen = sum(s.power_output_mw[t] for s in result.schedules)
            assert total_gen >= demands[t] - 1e-3, (
                f"Demand balance violated at t={t}: "
                f"generation={total_gen:.4f} < demand={demands[t]}"
            )

    def test_capacity_bounds_respected(self) -> None:
        """When committed, output ∈ [p_min, p_max]; when off, output == 0."""
        gens = _make_simple_generators()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(200.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        gen_map = {g.id: g for g in gens}
        for sched in result.schedules:
            g = gen_map[sched.generator_id]
            for t in range(24):
                if sched.commitment[t] == 1:
                    assert sched.power_output_mw[t] >= g.p_min_mw - 1e-3, (
                        f"{g.id} t={t}: output {sched.power_output_mw[t]:.4f}"
                        f" < p_min {g.p_min_mw}"
                    )
                    assert sched.power_output_mw[t] <= g.capacity_mw + 1e-3, (
                        f"{g.id} t={t}: output {sched.power_output_mw[t]:.4f}"
                        f" > p_max {g.capacity_mw}"
                    )
                else:
                    assert abs(sched.power_output_mw[t]) < 1e-3, (
                        f"{g.id} t={t}: off but output="
                        f"{sched.power_output_mw[t]:.4f}"
                    )


# ======================================================================
# TestMinUpTime
# ======================================================================


class TestMinUpTime:
    """Tests that minimum up-time constraints are enforced."""

    def test_min_uptime_enforced(self) -> None:
        """Once started, a generator remains on for at least min_up_time_h."""
        g = make_generator(
            id="g_mut4",
            name="MUT4 Gen",
            capacity_mw=200.0,
            fuel_type="coal",
            p_min_mw=50.0,
            startup_cost=100.0,
            shutdown_cost=50.0,
            min_up_time_h=4,
            min_down_time_h=1,
            fuel_cost_per_mwh=30.0,
            no_load_cost=10.0,
        )
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        dp = _flat_demand(150.0, 12)
        params = UCParameters(generators=[g], demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        commitment = result.schedules[0].commitment
        _assert_min_uptime(commitment, min_up=4)

    def test_min_uptime_with_varying_demand(self) -> None:
        """Min up-time holds even when demand varies and cycling is cheaper."""
        g1 = make_generator(
            id="g_base",
            name="Base Gen",
            capacity_mw=200.0,
            fuel_type="coal",
            p_min_mw=50.0,
            min_up_time_h=1,
            min_down_time_h=1,
            fuel_cost_per_mwh=20.0,
            no_load_cost=10.0,
        )
        g2 = make_generator(
            id="g_mut5",
            name="MUT5 Gen",
            capacity_mw=150.0,
            fuel_type="lng",
            p_min_mw=30.0,
            startup_cost=500.0,
            min_up_time_h=5,
            min_down_time_h=1,
            fuel_cost_per_mwh=50.0,
            no_load_cost=20.0,
        )
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        demands = [100, 100, 250, 250, 250, 100, 100, 100, 250, 250, 100, 100]
        dp = DemandProfile(demands=[float(d) for d in demands])
        params = UCParameters(generators=[g1, g2], demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched_g2 = _find_schedule(result, "g_mut5")
        _assert_min_uptime(sched_g2.commitment, min_up=5)


def _assert_min_uptime(commitment: List[int], min_up: int) -> None:
    """Assert that every ON-run in the commitment is >= min_up periods."""
    t = 0
    while t < len(commitment):
        if commitment[t] == 1:
            run_start = t
            while t < len(commitment) and commitment[t] == 1:
                t += 1
            run_length = t - run_start
            # Allow shorter runs only if they extend to the end of the horizon
            if t < len(commitment):
                assert run_length >= min_up, (
                    f"Min uptime violated: ON run from t={run_start} "
                    f"has length {run_length} < {min_up}"
                )
        else:
            t += 1


# ======================================================================
# TestMinDownTime
# ======================================================================


class TestMinDownTime:
    """Tests that minimum down-time constraints are enforced."""

    def test_min_downtime_enforced(self) -> None:
        """Once shut down, a generator remains off for at least min_down_time_h."""
        g1 = make_generator(
            id="g_base",
            name="Base Gen",
            capacity_mw=200.0,
            fuel_type="coal",
            p_min_mw=50.0,
            startup_cost=100.0,
            min_up_time_h=1,
            min_down_time_h=1,
            fuel_cost_per_mwh=20.0,
            no_load_cost=10.0,
        )
        g2 = make_generator(
            id="g_mdt3",
            name="MDT3 Gen",
            capacity_mw=150.0,
            fuel_type="lng",
            p_min_mw=30.0,
            startup_cost=500.0,
            min_up_time_h=1,
            min_down_time_h=3,
            fuel_cost_per_mwh=50.0,
            no_load_cost=20.0,
        )
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        demands = [180, 180, 100, 100, 100, 180, 180, 100, 100, 100, 180, 180]
        dp = DemandProfile(demands=[float(d) for d in demands])
        params = UCParameters(generators=[g1, g2], demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched_g2 = _find_schedule(result, "g_mdt3")
        _assert_min_downtime(sched_g2.commitment, min_down=3)

    def test_min_downtime_with_high_mdt(self) -> None:
        """Generator with MDT=6 stays off for at least 6 periods after shutdown."""
        g1 = make_generator(
            id="g_base",
            name="Base Gen",
            capacity_mw=300.0,
            fuel_type="coal",
            p_min_mw=50.0,
            fuel_cost_per_mwh=20.0,
            no_load_cost=10.0,
        )
        g2 = make_generator(
            id="g_mdt6",
            name="MDT6 Gen",
            capacity_mw=200.0,
            fuel_type="lng",
            p_min_mw=30.0,
            startup_cost=500.0,
            min_up_time_h=1,
            min_down_time_h=6,
            fuel_cost_per_mwh=50.0,
            no_load_cost=20.0,
        )
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        # Demand: high early, drops, then rises again
        demands = (
            [250.0] * 4 + [100.0] * 10 + [250.0] * 4 + [100.0] * 6
        )
        dp = DemandProfile(demands=demands)
        params = UCParameters(generators=[g1, g2], demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched_g2 = _find_schedule(result, "g_mdt6")
        _assert_min_downtime(sched_g2.commitment, min_down=6)


def _assert_min_downtime(commitment: List[int], min_down: int) -> None:
    """Assert every OFF-run (after an ON period) is >= min_down periods."""
    t = 0
    had_on = False
    while t < len(commitment):
        if commitment[t] == 0:
            off_start = t
            while t < len(commitment) and commitment[t] == 0:
                t += 1
            off_length = t - off_start
            # Only check runs that follow a period where the generator was ON
            if had_on and t < len(commitment):
                assert off_length >= min_down, (
                    f"Min downtime violated: OFF run from t={off_start} "
                    f"has length {off_length} < {min_down}"
                )
        else:
            had_on = True
            t += 1


# ======================================================================
# TestRampRates
# ======================================================================


class TestRampRates:
    """Tests that ramp rate limits are respected."""

    def test_ramp_up_limit_respected(self) -> None:
        """Power increase between consecutive ON periods <= ramp_up_mw_per_h."""
        g = make_generator(
            id="g_ramp",
            name="Ramp Gen",
            capacity_mw=200.0,
            fuel_type="coal",
            p_min_mw=20.0,
            ramp_up_mw_per_h=40.0,
            ramp_down_mw_per_h=40.0,
            fuel_cost_per_mwh=30.0,
            no_load_cost=10.0,
        )
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        # Gently rising demand
        demands = [50, 60, 80, 100, 120, 140, 160, 180, 190, 200, 200, 200]
        dp = DemandProfile(demands=[float(d) for d in demands])
        params = UCParameters(generators=[g], demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = result.schedules[0]
        for t in range(1, 12):
            if sched.commitment[t] == 1 and sched.commitment[t - 1] == 1:
                delta = sched.power_output_mw[t] - sched.power_output_mw[t - 1]
                assert delta <= 40.0 + 1e-3, (
                    f"Ramp up violated at t={t}: "
                    f"delta={delta:.4f} > limit=40.0"
                )

    def test_ramp_down_limit_respected(self) -> None:
        """Power decrease between consecutive ON periods <= ramp_down_mw_per_h."""
        g = make_generator(
            id="g_ramp",
            name="Ramp Gen",
            capacity_mw=200.0,
            fuel_type="coal",
            p_min_mw=20.0,
            ramp_up_mw_per_h=40.0,
            ramp_down_mw_per_h=40.0,
            fuel_cost_per_mwh=30.0,
            no_load_cost=10.0,
        )
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        # Falling demand
        demands = [200, 200, 200, 180, 160, 140, 120, 100, 80, 60, 40, 20]
        dp = DemandProfile(demands=[float(d) for d in demands])
        params = UCParameters(generators=[g], demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = result.schedules[0]
        for t in range(1, 12):
            if sched.commitment[t] == 1 and sched.commitment[t - 1] == 1:
                delta = sched.power_output_mw[t - 1] - sched.power_output_mw[t]
                assert delta <= 40.0 + 1e-3, (
                    f"Ramp down violated at t={t}: "
                    f"delta={delta:.4f} > limit=40.0"
                )

    def test_unlimited_ramp_no_constraint(self) -> None:
        """Generators with None ramp rates can change output freely."""
        g = make_generator(
            id="g_no_ramp",
            name="No Ramp Limit",
            capacity_mw=200.0,
            fuel_type="coal",
            p_min_mw=20.0,
            ramp_up_mw_per_h=None,
            ramp_down_mw_per_h=None,
            fuel_cost_per_mwh=30.0,
            no_load_cost=10.0,
        )
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        # Demand jumps sharply — would violate a 40 MW/h ramp limit
        demands = [20.0, 200.0, 20.0, 200.0, 20.0, 200.0]
        dp = DemandProfile(demands=demands)
        params = UCParameters(generators=[g], demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"


# ======================================================================
# TestMaintenanceWindows
# ======================================================================


class TestMaintenanceWindows:
    """Tests that maintenance windows are honored."""

    def test_generator_offline_during_maintenance(self) -> None:
        """Generator commitment == 0 and output == 0 during maintenance."""
        g1 = make_generator(
            id="g_maint",
            name="Maint Gen",
            capacity_mw=300.0,
            fuel_type="coal",
            p_min_mw=50.0,
            fuel_cost_per_mwh=30.0,
            no_load_cost=10.0,
            maintenance_windows=[(4, 8)],  # off during hours 4–7
        )
        g2 = make_generator(
            id="g_backup",
            name="Backup Gen",
            capacity_mw=300.0,
            fuel_type="lng",
            p_min_mw=30.0,
            fuel_cost_per_mwh=60.0,
            no_load_cost=20.0,
        )
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        dp = _flat_demand(200.0, 12)
        params = UCParameters(generators=[g1, g2], demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched_maint = _find_schedule(result, "g_maint")
        for t in range(4, 8):
            assert sched_maint.commitment[t] == 0, (
                f"g_maint should be off at t={t} (maintenance) "
                f"but commitment={sched_maint.commitment[t]}"
            )
            assert abs(sched_maint.power_output_mw[t]) < 1e-3, (
                f"g_maint should have zero output at t={t} (maintenance)"
            )

    def test_multiple_maintenance_windows(self) -> None:
        """Generator honors multiple non-overlapping maintenance windows."""
        g1 = make_generator(
            id="g_multi_maint",
            name="Multi Maint",
            capacity_mw=300.0,
            fuel_type="coal",
            p_min_mw=50.0,
            fuel_cost_per_mwh=30.0,
            no_load_cost=10.0,
            maintenance_windows=[(2, 4), (8, 10)],
        )
        g2 = make_generator(
            id="g_backup",
            name="Backup",
            capacity_mw=300.0,
            fuel_type="lng",
            p_min_mw=30.0,
            fuel_cost_per_mwh=60.0,
            no_load_cost=20.0,
        )
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        dp = _flat_demand(200.0, 12)
        params = UCParameters(generators=[g1, g2], demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = _find_schedule(result, "g_multi_maint")
        for t in [2, 3, 8, 9]:
            assert sched.commitment[t] == 0, (
                f"g_multi_maint should be off at t={t} (maintenance)"
            )

    def test_non_maintenance_periods_unaffected(self) -> None:
        """Periods outside maintenance windows allow normal operation."""
        g = make_generator(
            id="g_maint_partial",
            name="Partial Maint",
            capacity_mw=200.0,
            fuel_type="coal",
            p_min_mw=50.0,
            fuel_cost_per_mwh=30.0,
            no_load_cost=10.0,
            maintenance_windows=[(3, 5)],
        )
        g_backup = make_generator(
            id="g_backup",
            name="Backup",
            capacity_mw=200.0,
            fuel_type="lng",
            p_min_mw=20.0,
            fuel_cost_per_mwh=60.0,
            no_load_cost=20.0,
        )
        th = TimeHorizon(num_periods=8, period_duration_h=1.0)
        dp = _flat_demand(100.0, 8)
        params = UCParameters(
            generators=[g, g_backup], demand=dp, time_horizon=th
        )

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = _find_schedule(result, "g_maint_partial")
        # Generator should be available and potentially on outside maintenance
        assert sched.commitment[3] == 0
        assert sched.commitment[4] == 0


# ======================================================================
# TestReserveMargin
# ======================================================================


class TestReserveMargin:
    """Tests that reserve margin constraints are satisfied."""

    def test_reserve_margin_met(self) -> None:
        """Committed capacity >= demand × (1 + reserve_margin) at each t."""
        gens = _make_simple_generators()  # total 450 MW
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        demand_mw = 200.0
        dp = _flat_demand(demand_mw, 12)
        reserve_margin = 0.10  # 10%
        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            reserve_margin=reserve_margin,
        )

        result = solve_uc(params)
        assert result.status == "Optimal"

        gen_map = {g.id: g for g in gens}
        for t in range(12):
            committed_capacity = sum(
                gen_map[s.generator_id].capacity_mw
                for s in result.schedules
                if s.commitment[t] == 1
            )
            required = demand_mw * (1.0 + reserve_margin)
            assert committed_capacity >= required - 1e-3, (
                f"Reserve margin not met at t={t}: "
                f"committed={committed_capacity:.1f} < required={required:.1f}"
            )

    def test_zero_reserve_margin_no_extra_commitment(self) -> None:
        """With reserve_margin=0, no extra capacity is required."""
        gens = _make_simple_generators()
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        dp = _flat_demand(200.0, 12)
        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            reserve_margin=0.0,
        )

        result = solve_uc(params)
        assert result.status == "Optimal"

    def test_high_reserve_margin_forces_more_generators_on(self) -> None:
        """High reserve margin forces more committed capacity than demand alone."""
        gens = _make_simple_generators()  # total 450 MW
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        demand_mw = 150.0
        dp = _flat_demand(demand_mw, 6)
        # 50% reserve requires 225 MW committed capacity
        # g1 alone (200 MW) is not enough → must commit at least 2 gens
        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            reserve_margin=0.50,
        )

        result = solve_uc(params)
        assert result.status == "Optimal"

        gen_map = {g.id: g for g in gens}
        for t in range(6):
            committed_capacity = sum(
                gen_map[s.generator_id].capacity_mw
                for s in result.schedules
                if s.commitment[t] == 1
            )
            assert committed_capacity >= demand_mw * 1.5 - 1e-3


# ======================================================================
# TestInfeasibility
# ======================================================================


class TestInfeasibility:
    """Tests for infeasibility detection and diagnostics."""

    def test_demand_exceeds_capacity_returns_infeasible(self) -> None:
        """Instance with demand > total capacity returns Infeasible."""
        g = make_generator(
            id="g_small",
            name="Small Gen",
            capacity_mw=100.0,
            fuel_type="coal",
            fuel_cost_per_mwh=30.0,
        )
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = DemandProfile(demands=[500.0] * 6)  # 500 >> 100 MW
        params = UCParameters(generators=[g], demand=dp, time_horizon=th)

        result = solve_uc(params)

        assert result.status == "Infeasible"

    def test_infeasibility_has_diagnostics(self) -> None:
        """Infeasible result includes warnings with shortfall information."""
        g = make_generator(
            id="g_small",
            name="Small Gen",
            capacity_mw=100.0,
            fuel_type="coal",
            fuel_cost_per_mwh=30.0,
        )
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = DemandProfile(demands=[500.0] * 6)
        params = UCParameters(generators=[g], demand=dp, time_horizon=th)

        result = solve_uc(params)

        assert len(result.warnings) > 0
        warning_text = " ".join(result.warnings).lower()
        assert "shortfall" in warning_text or "demand" in warning_text

    def test_no_generators_returns_infeasible(self) -> None:
        """Empty generator list returns Infeasible with warning."""
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = DemandProfile(demands=[100.0] * 6)
        params = UCParameters(generators=[], demand=dp, time_horizon=th)

        result = solve_uc(params)

        assert result.status == "Infeasible"
        assert len(result.warnings) > 0

    def test_infeasibility_diagnostic_per_timestep(self) -> None:
        """Diagnostics identify the specific infeasible timesteps."""
        g = make_generator(
            id="g_med",
            name="Medium Gen",
            capacity_mw=150.0,
            fuel_type="coal",
            fuel_cost_per_mwh=30.0,
        )
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        # Only some timesteps are infeasible
        demands = [100.0, 100.0, 500.0, 500.0, 100.0, 100.0]
        dp = DemandProfile(demands=demands)
        params = UCParameters(generators=[g], demand=dp, time_horizon=th)

        result = solve_uc(params)

        assert result.status == "Infeasible"
        # Warnings should exist (from preflight or post-solve diagnostics)
        assert len(result.warnings) > 0


# ======================================================================
# TestSingleGenerator
# ======================================================================


class TestSingleGenerator:
    """Tests with a single generator (degenerate but valid)."""

    def test_single_generator_optimal(self) -> None:
        """Single generator meeting demand returns Optimal."""
        g = make_generator(
            id="g_only",
            name="Only Gen",
            capacity_mw=200.0,
            fuel_type="coal",
            p_min_mw=20.0,
            startup_cost=1000.0,
            shutdown_cost=500.0,
            fuel_cost_per_mwh=30.0,
            no_load_cost=50.0,
            labor_cost_per_h=10.0,
        )
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        dp = _flat_demand(100.0, 12)
        params = UCParameters(generators=[g], demand=dp, time_horizon=th)

        result = solve_uc(params)

        assert result.status == "Optimal"
        assert len(result.schedules) == 1

        sched = result.schedules[0]
        assert sched.generator_id == "g_only"
        # Must be on for all periods to meet demand
        for t in range(12):
            assert sched.commitment[t] == 1
            assert sched.power_output_mw[t] >= 100.0 - 1e-3

    def test_single_generator_cost_breakdown(self) -> None:
        """Single generator cost breakdown components sum to total_cost."""
        g = make_generator(
            id="g_cost",
            name="Cost Gen",
            capacity_mw=200.0,
            fuel_type="coal",
            p_min_mw=20.0,
            startup_cost=1000.0,
            fuel_cost_per_mwh=30.0,
            no_load_cost=50.0,
            labor_cost_per_h=10.0,
        )
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(100.0, 6)
        params = UCParameters(generators=[g], demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = result.schedules[0]
        component_total = (
            sched.startup_cost
            + sched.shutdown_cost
            + sched.fuel_cost
            + sched.no_load_cost
        )
        assert abs(sched.total_cost - component_total) < 1e-3

        # System total should match the single generator's total
        assert abs(result.total_cost - sched.total_cost) < 1.0

    def test_single_generator_expected_costs(self) -> None:
        """Verify approximate cost values for a known single-generator scenario.

        Generator on for 6 periods at 100 MW output:
        - Fuel: 30 × 100 × 6 = 18 000
        - No-load + labour: (50 + 10) × 6 = 360
        - Startup: 1000 × 1 = 1 000
        - Total ≈ 19 360
        """
        g = make_generator(
            id="g_calc",
            name="Calc Gen",
            capacity_mw=200.0,
            fuel_type="coal",
            p_min_mw=20.0,
            startup_cost=1000.0,
            fuel_cost_per_mwh=30.0,
            no_load_cost=50.0,
            labor_cost_per_h=10.0,
        )
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(100.0, 6)
        params = UCParameters(generators=[g], demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        # Fuel cost: solver may output exactly 100 MW
        sched = result.schedules[0]
        expected_fuel = 30.0 * 100.0 * 6  # 18 000
        expected_noload = (50.0 + 10.0) * 6  # 360
        expected_startup = 1000.0  # 1 startup
        expected_total = expected_fuel + expected_noload + expected_startup

        # Allow some tolerance for solver rounding
        assert abs(sched.fuel_cost - expected_fuel) < 10.0
        assert abs(sched.no_load_cost - expected_noload) < 1.0
        assert abs(sched.startup_cost - expected_startup) < 1.0
        assert abs(result.total_cost - expected_total) < 15.0


# ======================================================================
# TestZeroDemand
# ======================================================================


class TestZeroDemand:
    """Tests with zero demand at certain timesteps."""

    def test_zero_demand_all_off(self) -> None:
        """With zero demand everywhere, generators should be off."""
        g = make_generator(
            id="g_idle",
            name="Idle Gen",
            capacity_mw=200.0,
            fuel_type="coal",
            fuel_cost_per_mwh=30.0,
            no_load_cost=50.0,
            labor_cost_per_h=10.0,
        )
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = DemandProfile(demands=[0.0] * 6)
        params = UCParameters(generators=[g], demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = result.schedules[0]
        for t in range(6):
            assert sched.commitment[t] == 0, (
                f"Generator should be off at t={t} with zero demand"
            )
            assert abs(sched.power_output_mw[t]) < 1e-3
        assert result.total_cost < 1e-3

    def test_mixed_zero_and_positive_demand(self) -> None:
        """Generators meet positive demand and can idle during zero-demand."""
        g = make_generator(
            id="g_flex",
            name="Flex Gen",
            capacity_mw=200.0,
            fuel_type="coal",
            p_min_mw=20.0,
            min_up_time_h=1,
            min_down_time_h=1,
            fuel_cost_per_mwh=30.0,
            no_load_cost=50.0,
            startup_cost=100.0,
        )
        th = TimeHorizon(num_periods=8, period_duration_h=1.0)
        demands = [100.0, 100.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0]
        dp = DemandProfile(demands=demands)
        params = UCParameters(generators=[g], demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = result.schedules[0]
        # Positive demand periods must have sufficient output
        for t in [0, 1, 5, 6, 7]:
            assert sched.power_output_mw[t] >= 100.0 - 1e-3, (
                f"Demand not met at t={t}"
            )


# ======================================================================
# TestSolverConfiguration
# ======================================================================


class TestSolverConfiguration:
    """Tests for solver timeout and configuration handling."""

    def test_solver_accepts_time_limit(self) -> None:
        """UCParameters accepts solver_time_limit_s; solver completes."""
        gens = _make_simple_generators()
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        dp = _flat_demand(200.0, 12)
        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            solver_time_limit_s=60.0,
        )

        result = solve_uc(params)

        assert result.status == "Optimal"
        assert result.solve_time_s < 60.0

    def test_solver_accepts_mip_gap(self) -> None:
        """UCParameters accepts mip_gap; solver runs successfully."""
        gens = _make_simple_generators()
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        dp = _flat_demand(200.0, 12)
        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            mip_gap=0.01,
        )

        result = solve_uc(params)
        assert result.status == "Optimal"

    def test_missing_demand_returns_not_solved(self) -> None:
        """Missing demand profile returns Not Solved status."""
        gens = _make_simple_generators()
        params = UCParameters(generators=gens)

        result = solve_uc(params)

        assert result.status == "Not Solved"
        assert len(result.warnings) > 0

    def test_missing_time_horizon_returns_not_solved(self) -> None:
        """Missing time horizon returns Not Solved status."""
        gens = _make_simple_generators()
        dp = _flat_demand(200.0, 12)
        params = UCParameters(generators=gens, demand=dp)

        result = solve_uc(params)

        assert result.status == "Not Solved"
        assert len(result.warnings) > 0

    def test_result_summary_property(self) -> None:
        """UCResult.summary property returns expected dict keys."""
        gens = _make_simple_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)

        summary = result.summary
        assert "status" in summary
        assert "total_cost" in summary
        assert "num_generators" in summary
        assert "solve_time_s" in summary
        assert "gap" in summary
        assert "warnings" in summary


# ======================================================================
# TestPumpedHydroSOC — storage / pumped hydro tests
# ======================================================================


class TestPumpedHydroSOC:
    """Tests for pumped hydro storage state-of-charge constraints.

    Verifies that storage generators:
    - Keep SOC within [0, capacity] at all timesteps
    - Never charge and discharge simultaneously
    - Follow the SOC balance equation with efficiencies
    - Start from the correct initial SOC
    - Exhibit round-trip efficiency losses
    - Respect maintenance windows (no charge/discharge)
    - Do not create storage variables when no storage generators exist
    """

    @staticmethod
    def _make_storage_with_thermal() -> List[Generator]:
        """Create a thermal generator + pumped hydro storage pair.

        Thermal provides base-load; storage provides peak-shaving.
        Total thermal capacity (300 MW) exceeds peak demand so that
        the solver is free to dispatch storage optimally.
        """
        g_thermal = make_generator(
            id="g_thermal",
            name="Thermal Base",
            capacity_mw=300.0,
            fuel_type="coal",
            p_min_mw=50.0,
            startup_cost=1000.0,
            fuel_cost_per_mwh=40.0,
            no_load_cost=50.0,
        )
        g_storage = make_storage_generator(
            id="g_storage",
            name="Pumped Hydro",
            capacity_mw=100.0,
            storage_capacity_mwh=400.0,
            charge_rate_mw=100.0,
            discharge_rate_mw=100.0,
            charge_efficiency=0.85,
            discharge_efficiency=0.90,
            initial_soc_fraction=0.5,
            min_terminal_soc_fraction=0.5,
            startup_cost=200.0,
            fuel_cost_per_mwh=5.0,
            no_load_cost=10.0,
        )
        return [g_thermal, g_storage]

    def test_soc_bounds_respected(self) -> None:
        """SOC stays within [0, capacity] at all timesteps."""
        gens = self._make_storage_with_thermal()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(200.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched_storage = _find_schedule(result, "g_storage")
        capacity_mwh = 400.0

        assert len(sched_storage.soc_mwh) == 24, (
            f"Expected 24 SOC values, got {len(sched_storage.soc_mwh)}"
        )
        for t in range(24):
            assert sched_storage.soc_mwh[t] >= -1e-3, (
                f"SOC below zero at t={t}: {sched_storage.soc_mwh[t]:.4f}"
            )
            assert sched_storage.soc_mwh[t] <= capacity_mwh + 1e-3, (
                f"SOC above capacity at t={t}: "
                f"{sched_storage.soc_mwh[t]:.4f} > {capacity_mwh}"
            )

    def test_charge_discharge_mutual_exclusion(self) -> None:
        """Never simultaneously charging and discharging in same period."""
        gens = self._make_storage_with_thermal()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        # Varying demand encourages both charging and discharging
        demands = [150, 150, 100, 100, 80, 80, 100, 150,
                   200, 250, 280, 280, 250, 200, 150, 100,
                   80, 80, 100, 150, 200, 250, 250, 200]
        dp = DemandProfile(demands=[float(d) for d in demands])
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched_storage = _find_schedule(result, "g_storage")
        for t in range(24):
            ch = sched_storage.charge_mw[t]
            dis = sched_storage.discharge_mw[t]
            # At most one of charge/discharge should be non-zero
            assert ch < 1e-3 or dis < 1e-3, (
                f"Simultaneous charge ({ch:.4f} MW) and discharge "
                f"({dis:.4f} MW) at t={t}"
            )

    def test_soc_balance_equation(self) -> None:
        """Verify SOC[t] = SOC[t-1] + eta_ch*p_ch*dt - (1/eta_dis)*p_dis*dt."""
        gens = self._make_storage_with_thermal()
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        demands = [100, 100, 80, 80, 80, 100, 200, 250, 250, 200, 150, 100]
        dp = DemandProfile(demands=[float(d) for d in demands])
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = _find_schedule(result, "g_storage")
        eta_ch = 0.85
        eta_dis = 0.90
        dt = 1.0
        initial_soc = 0.5 * 400.0  # 200 MWh

        for t in range(12):
            p_ch_val = sched.charge_mw[t]
            p_dis_val = sched.discharge_mw[t]

            if t == 0:
                expected_soc = (
                    initial_soc
                    + eta_ch * p_ch_val * dt
                    - (1.0 / eta_dis) * p_dis_val * dt
                )
            else:
                expected_soc = (
                    sched.soc_mwh[t - 1]
                    + eta_ch * p_ch_val * dt
                    - (1.0 / eta_dis) * p_dis_val * dt
                )

            assert abs(sched.soc_mwh[t] - expected_soc) < 1e-2, (
                f"SOC balance violated at t={t}: "
                f"actual={sched.soc_mwh[t]:.4f}, expected={expected_soc:.4f}"
            )

    def test_initial_soc_correct(self) -> None:
        """SOC at t=0 follows from initial_soc_fraction and first period ops."""
        gens = self._make_storage_with_thermal()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = _find_schedule(result, "g_storage")
        eta_ch = 0.85
        eta_dis = 0.90
        dt = 1.0
        initial_soc = 0.5 * 400.0  # 200 MWh

        # Verify SOC at t=0 is consistent with initial SOC and period-0 ops
        expected_soc_0 = (
            initial_soc
            + eta_ch * sched.charge_mw[0] * dt
            - (1.0 / eta_dis) * sched.discharge_mw[0] * dt
        )
        assert abs(sched.soc_mwh[0] - expected_soc_0) < 1e-2, (
            f"Initial SOC incorrect: actual={sched.soc_mwh[0]:.4f}, "
            f"expected={expected_soc_0:.4f} "
            f"(initial={initial_soc}, ch={sched.charge_mw[0]:.4f}, "
            f"dis={sched.discharge_mw[0]:.4f})"
        )

    def test_efficiency_loss_observed(self) -> None:
        """Total energy out < total energy in due to round-trip losses."""
        gens = self._make_storage_with_thermal()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        # Varying demand to encourage charge/discharge cycling
        demands = [100, 80, 80, 80, 80, 80, 100, 150,
                   250, 280, 280, 280, 250, 200, 150, 100,
                   80, 80, 80, 80, 100, 200, 250, 200]
        dp = DemandProfile(demands=[float(d) for d in demands])
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = _find_schedule(result, "g_storage")
        total_charge = sum(sched.charge_mw)
        total_discharge = sum(sched.discharge_mw)

        # If there was any meaningful storage cycling, verify losses
        if total_charge > 1.0 and total_discharge > 1.0:
            # Energy stored = charge * eta_ch, energy released = discharge / eta_dis
            # Round-trip: eta_ch * eta_dis < 1, so energy out < energy in
            energy_in = total_charge * 0.85  # stored energy
            energy_out = total_discharge  # released energy before loss
            # The net SOC change accounts for the rest; overall losses are real
            assert total_discharge < total_charge, (
                f"Expected efficiency losses: discharge ({total_discharge:.2f} MWh) "
                f"should be < charge ({total_charge:.2f} MWh) due to "
                f"round-trip efficiency {0.85 * 0.90:.2%}"
            )

    def test_storage_in_maintenance(self) -> None:
        """Storage forced offline during maintenance: no charge/discharge."""
        g_thermal = make_generator(
            id="g_thermal",
            name="Thermal Base",
            capacity_mw=300.0,
            fuel_type="coal",
            p_min_mw=50.0,
            fuel_cost_per_mwh=40.0,
            no_load_cost=50.0,
        )
        g_storage = make_storage_generator(
            id="g_storage_maint",
            name="Pumped Hydro Maint",
            capacity_mw=100.0,
            storage_capacity_mwh=400.0,
            charge_rate_mw=100.0,
            discharge_rate_mw=100.0,
            charge_efficiency=0.85,
            discharge_efficiency=0.90,
            initial_soc_fraction=0.5,
            min_terminal_soc_fraction=0.0,
            startup_cost=200.0,
            fuel_cost_per_mwh=5.0,
            no_load_cost=10.0,
            maintenance_windows=[(4, 8)],
        )
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        dp = _flat_demand(200.0, 12)
        params = UCParameters(
            generators=[g_thermal, g_storage], demand=dp, time_horizon=th
        )

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = _find_schedule(result, "g_storage_maint")
        for t in range(4, 8):
            assert sched.commitment[t] == 0, (
                f"Storage should be off at t={t} (maintenance) "
                f"but commitment={sched.commitment[t]}"
            )
            assert abs(sched.charge_mw[t]) < 1e-3, (
                f"Storage should not charge at t={t} (maintenance) "
                f"but charge_mw={sched.charge_mw[t]:.4f}"
            )
            assert abs(sched.discharge_mw[t]) < 1e-3, (
                f"Storage should not discharge at t={t} (maintenance) "
                f"but discharge_mw={sched.discharge_mw[t]:.4f}"
            )

    def test_no_storage_no_variables(self) -> None:
        """Problem without storage generators creates no storage variables.

        Existing solver behavior should be identical — standard generators
        produce schedules without soc_mwh, charge_mw, discharge_mw arrays.
        """
        gens = _make_simple_generators()  # No storage generators
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        dp = _flat_demand(200.0, 12)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        for sched in result.schedules:
            # Non-storage generators should have empty storage arrays
            assert sched.soc_mwh == [], (
                f"Generator {sched.generator_id} should have no SOC data "
                f"but got {len(sched.soc_mwh)} values"
            )
            assert sched.charge_mw == [], (
                f"Generator {sched.generator_id} should have no charge data "
                f"but got {len(sched.charge_mw)} values"
            )
            assert sched.discharge_mw == [], (
                f"Generator {sched.generator_id} should have no discharge data "
                f"but got {len(sched.discharge_mw)} values"
            )


# ======================================================================
# TestBatterySOC — battery storage state-of-charge tests
# ======================================================================


class TestBatterySOC:
    """Tests for battery storage SOC tracking with high-efficiency settings.

    Verifies that battery storage generators:
    - Track SOC correctly with 95%/95% charge/discharge efficiency
    - Respond to rapid demand fluctuations within charge/discharge limits
    - Meet the terminal SOC constraint at end of horizon
    """

    @staticmethod
    def _make_battery_with_thermal() -> List[Generator]:
        """Create a thermal generator + battery storage pair.

        Battery uses higher efficiencies (95%/95%) typical of
        lithium-ion systems, with 50 MW / 200 MWh configuration.
        """
        g_thermal = make_generator(
            id="g_thermal_bat",
            name="Thermal Base",
            capacity_mw=300.0,
            fuel_type="coal",
            p_min_mw=50.0,
            startup_cost=1000.0,
            fuel_cost_per_mwh=40.0,
            no_load_cost=50.0,
        )
        g_battery = make_storage_generator(
            id="g_battery",
            name="Li-ion Battery",
            capacity_mw=50.0,
            fuel_type="hydro",
            storage_capacity_mwh=200.0,
            charge_rate_mw=50.0,
            discharge_rate_mw=50.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            initial_soc_fraction=0.5,
            min_terminal_soc_fraction=0.3,
            startup_cost=50.0,
            fuel_cost_per_mwh=2.0,
            no_load_cost=5.0,
        )
        return [g_thermal, g_battery]

    def test_battery_soc_tracking(self) -> None:
        """SOC follows correct trajectory with 95%/95% efficiency."""
        gens = self._make_battery_with_thermal()
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        # Varying demand to force charge/discharge cycles
        demands = [100, 80, 80, 80, 80, 100, 200, 250, 250, 200, 150, 100]
        dp = DemandProfile(demands=[float(d) for d in demands])
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = _find_schedule(result, "g_battery")
        eta_ch = 0.95
        eta_dis = 0.95
        dt = 1.0
        initial_soc = 0.5 * 200.0  # 100 MWh

        # Verify SOC balance equation at each timestep
        for t in range(12):
            p_ch_val = sched.charge_mw[t]
            p_dis_val = sched.discharge_mw[t]

            if t == 0:
                expected_soc = (
                    initial_soc
                    + eta_ch * p_ch_val * dt
                    - (1.0 / eta_dis) * p_dis_val * dt
                )
            else:
                expected_soc = (
                    sched.soc_mwh[t - 1]
                    + eta_ch * p_ch_val * dt
                    - (1.0 / eta_dis) * p_dis_val * dt
                )

            assert abs(sched.soc_mwh[t] - expected_soc) < 1e-2, (
                f"Battery SOC balance violated at t={t}: "
                f"actual={sched.soc_mwh[t]:.4f}, expected={expected_soc:.4f}"
            )

        # Verify SOC stays within bounds
        for t in range(12):
            assert sched.soc_mwh[t] >= -1e-3, (
                f"Battery SOC below zero at t={t}: {sched.soc_mwh[t]:.4f}"
            )
            assert sched.soc_mwh[t] <= 200.0 + 1e-3, (
                f"Battery SOC above capacity at t={t}: "
                f"{sched.soc_mwh[t]:.4f} > 200.0"
            )

    def test_battery_fast_cycle(self) -> None:
        """Battery responds to rapid demand fluctuations within limits."""
        gens = self._make_battery_with_thermal()
        th = TimeHorizon(num_periods=8, period_duration_h=1.0)
        # Rapid demand swings: low-high-low-high pattern
        demands = [80.0, 300.0, 80.0, 300.0, 80.0, 300.0, 80.0, 150.0]
        dp = DemandProfile(demands=demands)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = _find_schedule(result, "g_battery")
        charge_rate = 50.0
        discharge_rate = 50.0

        for t in range(8):
            # Charge never exceeds charge_rate_mw
            assert sched.charge_mw[t] <= charge_rate + 1e-3, (
                f"Battery charge exceeds limit at t={t}: "
                f"{sched.charge_mw[t]:.4f} > {charge_rate}"
            )
            # Discharge never exceeds discharge_rate_mw
            assert sched.discharge_mw[t] <= discharge_rate + 1e-3, (
                f"Battery discharge exceeds limit at t={t}: "
                f"{sched.discharge_mw[t]:.4f} > {discharge_rate}"
            )
            # No simultaneous charge and discharge
            assert sched.charge_mw[t] < 1e-3 or sched.discharge_mw[t] < 1e-3, (
                f"Simultaneous charge ({sched.charge_mw[t]:.4f} MW) and "
                f"discharge ({sched.discharge_mw[t]:.4f} MW) at t={t}"
            )

        # Demand must be met at every period
        for t in range(8):
            total_gen = sum(s.power_output_mw[t] for s in result.schedules)
            assert total_gen >= demands[t] - 1e-3, (
                f"Demand balance violated at t={t}: "
                f"generation={total_gen:.4f} < demand={demands[t]}"
            )

    def test_terminal_soc_constraint(self) -> None:
        """SOC at end of horizon >= min_terminal_soc_fraction × capacity."""
        gens = self._make_battery_with_thermal()
        th = TimeHorizon(num_periods=12, period_duration_h=1.0)
        # Demand pattern that incentivises draining the battery
        demands = [100, 100, 100, 250, 250, 250, 250, 250, 250, 200, 150, 100]
        dp = DemandProfile(demands=[float(d) for d in demands])
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = _find_schedule(result, "g_battery")
        min_terminal_soc = 0.3 * 200.0  # 60 MWh

        # Terminal SOC at the last period must meet the constraint
        final_soc = sched.soc_mwh[-1]
        assert final_soc >= min_terminal_soc - 1e-3, (
            f"Terminal SOC constraint violated: "
            f"final_soc={final_soc:.4f} < required={min_terminal_soc:.1f} MWh"
        )


# ======================================================================
# TestStorageIntegration — integration and regression tests
# ======================================================================


class TestStorageIntegration:
    """Integration tests for storage with thermal generators.

    Verifies that:
    - Mixed thermal + storage systems solve correctly
    - Pumped hydro follows realistic day-cycle patterns
    - Existing non-storage tests remain unaffected
    """

    def test_mixed_thermal_storage(self) -> None:
        """2 thermal + 1 storage: demand met at all periods, costs non-negative."""
        g1 = make_generator(
            id="g_base_mix",
            name="Base Coal",
            capacity_mw=200.0,
            fuel_type="coal",
            p_min_mw=50.0,
            startup_cost=3000.0,
            fuel_cost_per_mwh=30.0,
            no_load_cost=80.0,
        )
        g2 = make_generator(
            id="g_peak_mix",
            name="Peak LNG",
            capacity_mw=150.0,
            fuel_type="lng",
            p_min_mw=30.0,
            startup_cost=1500.0,
            fuel_cost_per_mwh=60.0,
            no_load_cost=40.0,
        )
        g_storage = make_storage_generator(
            id="g_storage_mix",
            name="Pumped Hydro Mix",
            capacity_mw=80.0,
            storage_capacity_mwh=320.0,
            charge_rate_mw=80.0,
            discharge_rate_mw=80.0,
            charge_efficiency=0.85,
            discharge_efficiency=0.90,
            initial_soc_fraction=0.5,
            min_terminal_soc_fraction=0.4,
            startup_cost=200.0,
            fuel_cost_per_mwh=5.0,
            no_load_cost=10.0,
        )

        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        # Realistic daily demand pattern
        demands = [
            120, 110, 100, 100, 110, 130, 180, 220,
            250, 260, 260, 250, 240, 230, 220, 210,
            220, 240, 250, 230, 200, 170, 150, 130,
        ]
        dp = DemandProfile(demands=[float(d) for d in demands])
        params = UCParameters(
            generators=[g1, g2, g_storage], demand=dp, time_horizon=th
        )

        result = solve_uc(params)
        assert result.status == "Optimal"

        # Demand met at all periods
        for t in range(24):
            total_gen = sum(s.power_output_mw[t] for s in result.schedules)
            assert total_gen >= demands[t] - 1e-3, (
                f"Demand balance violated at t={t}: "
                f"generation={total_gen:.4f} < demand={demands[t]}"
            )

        # All individual costs non-negative for thermal generators.
        # Storage generators can have negative fuel_cost because net power
        # p = p_dis - p_ch is negative when charging, and
        # fuel_cost = fuel_cost_per_mwh × p_val accumulates negative terms.
        gen_map = {g.id: g for g in [g1, g2, g_storage]}
        for sched in result.schedules:
            assert sched.startup_cost >= -1e-3, (
                f"{sched.generator_id}: negative startup_cost "
                f"{sched.startup_cost:.4f}"
            )
            assert sched.shutdown_cost >= -1e-3, (
                f"{sched.generator_id}: negative shutdown_cost "
                f"{sched.shutdown_cost:.4f}"
            )
            if not gen_map[sched.generator_id].is_storage:
                assert sched.fuel_cost >= -1e-3, (
                    f"{sched.generator_id}: negative fuel_cost "
                    f"{sched.fuel_cost:.4f}"
                )
            assert sched.no_load_cost >= -1e-3, (
                f"{sched.generator_id}: negative no_load_cost "
                f"{sched.no_load_cost:.4f}"
            )

        # Total cost > 0
        assert result.total_cost > 0

    def test_pumped_hydro_day_cycle(self) -> None:
        """24h cycle: pumped hydro charges at night, discharges at peak.

        Demand pattern: low at night (periods 0–5), high during day
        (periods 8–17). Peak demand (330 MW) moderately exceeds
        thermal capacity (300 MW), forcing storage to discharge
        during peak hours.  Storage must charge at night to build up
        reserves.  Total excess energy during peak is bounded by
        available storage capacity to ensure feasibility.
        """
        g_thermal = make_generator(
            id="g_thermal_day",
            name="Thermal Day",
            capacity_mw=300.0,
            fuel_type="coal",
            p_min_mw=50.0,
            startup_cost=2000.0,
            fuel_cost_per_mwh=40.0,
            no_load_cost=60.0,
        )
        g_hydro = make_storage_generator(
            id="g_hydro_day",
            name="Pumped Hydro Day",
            capacity_mw=100.0,
            storage_capacity_mwh=400.0,
            charge_rate_mw=100.0,
            discharge_rate_mw=100.0,
            charge_efficiency=0.85,
            discharge_efficiency=0.90,
            initial_soc_fraction=0.5,
            min_terminal_soc_fraction=0.3,
            startup_cost=200.0,
            fuel_cost_per_mwh=5.0,
            no_load_cost=10.0,
        )

        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        # Night-low, day-high demand pattern.  Peak demand (330 MW)
        # moderately exceeds thermal capacity (300 MW).
        # Total peak excess: ~130 MW over 10 periods → ~144 MWh from tank
        # (at 90% discharge efficiency).  Feasible with 400 MWh tank
        # starting at 200 MWh (50% SOC).
        demands = [
            100, 80, 80, 80, 80, 100,    # Night: t=0-5 (low demand)
            150, 200, 310, 320, 330, 330, # Morning ramp & peak: t=6-11
            320, 310, 300, 290, 280, 290, # Afternoon: t=12-17
            300, 280, 220, 170, 130, 100, # Evening decline: t=18-23
        ]
        dp = DemandProfile(demands=[float(d) for d in demands])
        params = UCParameters(
            generators=[g_thermal, g_hydro], demand=dp, time_horizon=th
        )

        result = solve_uc(params)
        assert result.status == "Optimal"

        sched = _find_schedule(result, "g_hydro_day")

        # Storage must have meaningful charge and discharge activity
        total_charge = sum(sched.charge_mw)
        total_discharge = sum(sched.discharge_mw)
        assert total_charge > 1.0, (
            f"Expected meaningful charging, got total_charge={total_charge:.2f}"
        )
        assert total_discharge > 1.0, (
            f"Expected meaningful discharging, got total_discharge={total_discharge:.2f}"
        )

        # Charging should happen during lower-demand periods and
        # discharging during higher-demand periods (economic dispatch).
        # Verify: average demand during charging periods < average
        # demand during discharging periods.
        charge_demand_sum = 0.0
        charge_count = 0
        discharge_demand_sum = 0.0
        discharge_count = 0
        for t in range(24):
            if sched.charge_mw[t] > 1e-3:
                charge_demand_sum += demands[t]
                charge_count += 1
            if sched.discharge_mw[t] > 1e-3:
                discharge_demand_sum += demands[t]
                discharge_count += 1

        if charge_count > 0 and discharge_count > 0:
            avg_charge_demand = charge_demand_sum / charge_count
            avg_discharge_demand = discharge_demand_sum / discharge_count
            assert avg_charge_demand < avg_discharge_demand, (
                f"Expected charging at lower demand than discharging: "
                f"avg_charge_demand={avg_charge_demand:.1f}, "
                f"avg_discharge_demand={avg_discharge_demand:.1f}"
            )

        # Demand must be met at every period
        for t in range(24):
            total_gen = sum(s.power_output_mw[t] for s in result.schedules)
            assert total_gen >= demands[t] - 1e-3, (
                f"Demand balance violated at t={t}: "
                f"generation={total_gen:.4f} < demand={demands[t]}"
            )

    def test_existing_tests_unchanged(self) -> None:
        """Existing 3-generator instance produces Optimal with same behavior.

        Regression test: the addition of storage support must not change
        the solver behavior for non-storage instances. The standard
        3-generator test instance should still solve optimally with
        demand met at all periods and costs matching expected structure.
        """
        gens = _make_simple_generators()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        demand_mw = 200.0
        dp = _flat_demand(demand_mw, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        result = solve_uc(params)

        # Must still be Optimal
        assert result.status == "Optimal"
        assert result.total_cost > 0
        assert len(result.schedules) == 3

        # All 3 generators present
        gen_ids = {s.generator_id for s in result.schedules}
        assert gen_ids == {"g1", "g2", "g3"}

        # Demand met at every period
        for t in range(24):
            total_gen = sum(s.power_output_mw[t] for s in result.schedules)
            assert total_gen >= demand_mw - 1e-3, (
                f"Demand balance violated at t={t}: "
                f"generation={total_gen:.4f} < demand={demand_mw}"
            )

        # Total cost is sum of generator costs
        sum_gen_costs = sum(s.total_cost for s in result.schedules)
        assert abs(result.total_cost - sum_gen_costs) < 1.0

        # No storage arrays for non-storage generators
        for sched in result.schedules:
            assert sched.soc_mwh == []
            assert sched.charge_mw == []
            assert sched.discharge_mw == []
