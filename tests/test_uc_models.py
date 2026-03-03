"""Unit tests for unit commitment (UC) data models.

Tests for Generator UC fields (backward compatibility, all new fields,
validation of invalid values), TimeHorizon, DemandProfile, UCParameters,
UCResult with summary property, and GeneratorSchedule.
"""

import pytest

from src.model.generator import Generator
from src.uc.models import (
    DemandProfile,
    GeneratorSchedule,
    TimeHorizon,
    UCParameters,
    UCResult,
)

# Import the factory function for creating test generators
from tests.conftest import make_generator


# ======================================================================
# Generator: backward compatibility with UC fields
# ======================================================================


class TestGeneratorBackwardCompatibility:
    """Verify that Generator still works without any UC fields specified."""

    def test_default_uc_fields(self) -> None:
        """Generator created without UC kwargs uses safe defaults."""
        gen = Generator(
            id="gen_001",
            name="TestPlant",
            capacity_mw=100.0,
            fuel_type="coal",
        )
        assert gen.startup_cost == 0.0
        assert gen.shutdown_cost == 0.0
        assert gen.min_up_time_h == 1
        assert gen.min_down_time_h == 1
        assert gen.ramp_up_mw_per_h is None
        assert gen.ramp_down_mw_per_h is None
        assert gen.fuel_cost_per_mwh == 0.0
        assert gen.labor_cost_per_h == 0.0
        assert gen.no_load_cost == 0.0
        assert gen.maintenance_windows == []
        assert gen.construction_date is None
        assert gen.rebuild_planned_date is None
        assert gen.disaster_risk_score == 0.0

    def test_make_generator_without_uc_kwargs(self) -> None:
        """make_generator() factory works without UC kwargs."""
        gen = make_generator()
        assert gen.id == "shikoku_gen_001"
        assert gen.capacity_mw == 500.0
        assert gen.startup_cost == 0.0
        assert gen.maintenance_windows == []

    def test_existing_properties_unchanged(self) -> None:
        """Existing properties (has_location, is_connected, etc.) still work."""
        gen = make_generator(
            latitude=33.8,
            longitude=133.5,
            connected_bus_id="sub_001",
        )
        assert gen.has_location is True
        assert gen.is_connected is True
        assert gen.geodata == (133.5, 33.8)


# ======================================================================
# Generator: UC-specific fields
# ======================================================================


class TestGeneratorUCFields:
    """Tests for Generator with all UC-specific fields populated."""

    def test_all_uc_fields(self) -> None:
        """Generator accepts all UC fields via constructor."""
        gen = make_generator(
            startup_cost=50000.0,
            shutdown_cost=10000.0,
            min_up_time_h=4,
            min_down_time_h=2,
            ramp_up_mw_per_h=100.0,
            ramp_down_mw_per_h=80.0,
            fuel_cost_per_mwh=25.5,
            labor_cost_per_h=1500.0,
            no_load_cost=3000.0,
            maintenance_windows=[(10, 20), (100, 110)],
            construction_date="1990-04-01",
            rebuild_planned_date="2030-01-15",
            disaster_risk_score=0.35,
        )
        assert gen.startup_cost == 50000.0
        assert gen.shutdown_cost == 10000.0
        assert gen.min_up_time_h == 4
        assert gen.min_down_time_h == 2
        assert gen.ramp_up_mw_per_h == 100.0
        assert gen.ramp_down_mw_per_h == 80.0
        assert gen.fuel_cost_per_mwh == 25.5
        assert gen.labor_cost_per_h == 1500.0
        assert gen.no_load_cost == 3000.0
        assert gen.maintenance_windows == [(10, 20), (100, 110)]
        assert gen.construction_date == "1990-04-01"
        assert gen.rebuild_planned_date == "2030-01-15"
        assert gen.disaster_risk_score == 0.35

    def test_ramp_rates_none_means_unlimited(self) -> None:
        """None ramp rates indicate unlimited ramping capability."""
        gen = make_generator(ramp_up_mw_per_h=None, ramp_down_mw_per_h=None)
        assert gen.ramp_up_mw_per_h is None
        assert gen.ramp_down_mw_per_h is None

    def test_ramp_rates_zero_is_valid(self) -> None:
        """Zero ramp rates are accepted (generator cannot change output)."""
        gen = make_generator(ramp_up_mw_per_h=0.0, ramp_down_mw_per_h=0.0)
        assert gen.ramp_up_mw_per_h == 0.0
        assert gen.ramp_down_mw_per_h == 0.0

    def test_empty_maintenance_windows(self) -> None:
        """Empty maintenance windows list is valid."""
        gen = make_generator(maintenance_windows=[])
        assert gen.maintenance_windows == []

    def test_multiple_maintenance_windows(self) -> None:
        """Multiple maintenance windows are stored correctly."""
        windows = [(0, 5), (48, 72), (168, 192)]
        gen = make_generator(maintenance_windows=windows)
        assert gen.maintenance_windows == windows
        assert len(gen.maintenance_windows) == 3

    def test_is_renewable_with_uc_fields(self) -> None:
        """is_renewable property works correctly with UC fields set."""
        gen_solar = make_generator(
            fuel_type="solar",
            startup_cost=0.0,
            fuel_cost_per_mwh=0.0,
        )
        assert gen_solar.is_renewable is True

        gen_coal = make_generator(
            fuel_type="coal",
            startup_cost=50000.0,
            fuel_cost_per_mwh=25.0,
        )
        assert gen_coal.is_renewable is False


# ======================================================================
# Generator: validation of invalid UC values
# ======================================================================


class TestGeneratorUCValidation:
    """Tests for Generator validation of invalid UC field values."""

    def test_negative_startup_cost_raises(self) -> None:
        """Negative startup_cost raises ValueError."""
        with pytest.raises(ValueError, match="startup_cost"):
            make_generator(startup_cost=-1.0)

    def test_negative_shutdown_cost_raises(self) -> None:
        """Negative shutdown_cost raises ValueError."""
        with pytest.raises(ValueError, match="shutdown_cost"):
            make_generator(shutdown_cost=-100.0)

    def test_min_up_time_zero_raises(self) -> None:
        """min_up_time_h < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_up_time_h"):
            make_generator(min_up_time_h=0)

    def test_min_down_time_zero_raises(self) -> None:
        """min_down_time_h < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_down_time_h"):
            make_generator(min_down_time_h=0)

    def test_negative_ramp_up_raises(self) -> None:
        """Negative ramp_up_mw_per_h raises ValueError."""
        with pytest.raises(ValueError, match="ramp_up_mw_per_h"):
            make_generator(ramp_up_mw_per_h=-10.0)

    def test_negative_ramp_down_raises(self) -> None:
        """Negative ramp_down_mw_per_h raises ValueError."""
        with pytest.raises(ValueError, match="ramp_down_mw_per_h"):
            make_generator(ramp_down_mw_per_h=-5.0)

    def test_negative_fuel_cost_raises(self) -> None:
        """Negative fuel_cost_per_mwh raises ValueError."""
        with pytest.raises(ValueError, match="fuel_cost_per_mwh"):
            make_generator(fuel_cost_per_mwh=-1.0)

    def test_negative_labor_cost_raises(self) -> None:
        """Negative labor_cost_per_h raises ValueError."""
        with pytest.raises(ValueError, match="labor_cost_per_h"):
            make_generator(labor_cost_per_h=-500.0)

    def test_negative_no_load_cost_raises(self) -> None:
        """Negative no_load_cost raises ValueError."""
        with pytest.raises(ValueError, match="no_load_cost"):
            make_generator(no_load_cost=-1.0)

    def test_negative_disaster_risk_score_raises(self) -> None:
        """Negative disaster_risk_score raises ValueError."""
        with pytest.raises(ValueError, match="disaster_risk_score"):
            make_generator(disaster_risk_score=-0.1)

    def test_invalid_maintenance_window_order_raises(self) -> None:
        """Maintenance window with start >= end raises ValueError."""
        with pytest.raises(ValueError, match="start must be < end"):
            make_generator(maintenance_windows=[(10, 10)])

    def test_invalid_maintenance_window_reversed_raises(self) -> None:
        """Maintenance window with start > end raises ValueError."""
        with pytest.raises(ValueError, match="start must be < end"):
            make_generator(maintenance_windows=[(20, 10)])


# ======================================================================
# TimeHorizon
# ======================================================================


class TestTimeHorizon:
    """Tests for TimeHorizon dataclass."""

    def test_default_construction(self) -> None:
        """TimeHorizon with minimal args uses correct defaults."""
        th = TimeHorizon(num_periods=24)
        assert th.num_periods == 24
        assert th.period_duration_h == 1.0
        assert th.start_period == 0

    def test_custom_construction(self) -> None:
        """TimeHorizon accepts custom period duration and start."""
        th = TimeHorizon(num_periods=48, period_duration_h=0.5, start_period=10)
        assert th.num_periods == 48
        assert th.period_duration_h == 0.5
        assert th.start_period == 10

    def test_total_hours_hourly(self) -> None:
        """total_hours computes correctly for 1-hour intervals."""
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        assert th.total_hours == 24.0

    def test_total_hours_half_hourly(self) -> None:
        """total_hours computes correctly for 30-minute intervals."""
        th = TimeHorizon(num_periods=48, period_duration_h=0.5)
        assert th.total_hours == 24.0

    def test_period_indices_default_start(self) -> None:
        """period_indices returns [0, 1, ..., n-1] when start_period=0."""
        th = TimeHorizon(num_periods=4)
        assert th.period_indices == [0, 1, 2, 3]

    def test_period_indices_with_offset(self) -> None:
        """period_indices returns offset range when start_period > 0."""
        th = TimeHorizon(num_periods=3, start_period=5)
        assert th.period_indices == [5, 6, 7]

    def test_single_period(self) -> None:
        """Single-period horizon is valid."""
        th = TimeHorizon(num_periods=1, period_duration_h=4.0)
        assert th.total_hours == 4.0
        assert th.period_indices == [0]

    def test_zero_periods_raises(self) -> None:
        """num_periods=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_periods must be >= 1"):
            TimeHorizon(num_periods=0)

    def test_negative_periods_raises(self) -> None:
        """Negative num_periods raises ValueError."""
        with pytest.raises(ValueError, match="num_periods must be >= 1"):
            TimeHorizon(num_periods=-1)

    def test_zero_duration_raises(self) -> None:
        """period_duration_h=0 raises ValueError."""
        with pytest.raises(ValueError, match="period_duration_h must be positive"):
            TimeHorizon(num_periods=24, period_duration_h=0.0)

    def test_negative_duration_raises(self) -> None:
        """Negative period_duration_h raises ValueError."""
        with pytest.raises(ValueError, match="period_duration_h must be positive"):
            TimeHorizon(num_periods=24, period_duration_h=-1.0)

    def test_negative_start_period_raises(self) -> None:
        """Negative start_period raises ValueError."""
        with pytest.raises(ValueError, match="start_period must be non-negative"):
            TimeHorizon(num_periods=24, start_period=-1)


# ======================================================================
# DemandProfile
# ======================================================================


class TestDemandProfile:
    """Tests for DemandProfile dataclass."""

    def test_default_construction(self) -> None:
        """DemandProfile without args creates empty demand list."""
        dp = DemandProfile()
        assert dp.demands == []

    def test_construction_with_demands(self) -> None:
        """DemandProfile stores demands correctly."""
        demands = [100.0, 150.0, 200.0, 180.0]
        dp = DemandProfile(demands=demands)
        assert dp.demands == demands

    def test_peak_demand(self) -> None:
        """peak_demand returns the maximum demand value."""
        dp = DemandProfile(demands=[100.0, 250.0, 150.0, 200.0])
        assert dp.peak_demand == 250.0

    def test_peak_demand_empty(self) -> None:
        """peak_demand returns 0.0 for empty demand list."""
        dp = DemandProfile()
        assert dp.peak_demand == 0.0

    def test_total_energy_mwh(self) -> None:
        """total_energy_mwh returns sum of all demands."""
        dp = DemandProfile(demands=[100.0, 150.0, 200.0])
        assert dp.total_energy_mwh == 450.0

    def test_total_energy_empty(self) -> None:
        """total_energy_mwh returns 0 for empty demand list."""
        dp = DemandProfile()
        assert dp.total_energy_mwh == 0.0

    def test_uniform_demand(self) -> None:
        """Uniform demand profile has peak equal to any value."""
        dp = DemandProfile(demands=[100.0] * 24)
        assert dp.peak_demand == 100.0
        assert dp.total_energy_mwh == 2400.0

    def test_zero_demand_values_valid(self) -> None:
        """Zero demand values are accepted."""
        dp = DemandProfile(demands=[0.0, 100.0, 0.0])
        assert dp.peak_demand == 100.0

    def test_negative_demand_raises(self) -> None:
        """Negative demand values raise ValueError."""
        with pytest.raises(ValueError, match="Demand at period 1 must be non-negative"):
            DemandProfile(demands=[100.0, -50.0, 200.0])

    def test_negative_demand_first_period_raises(self) -> None:
        """Negative demand at period 0 raises ValueError."""
        with pytest.raises(ValueError, match="Demand at period 0"):
            DemandProfile(demands=[-10.0])


# ======================================================================
# UCParameters
# ======================================================================


class TestUCParameters:
    """Tests for UCParameters dataclass."""

    def test_default_construction(self) -> None:
        """UCParameters with defaults is valid."""
        params = UCParameters()
        assert params.generators == []
        assert params.demand is None
        assert params.time_horizon is None
        assert params.reserve_margin == 0.0
        assert params.solver_name == "HiGHS"
        assert params.solver_time_limit_s is None
        assert params.mip_gap is None
        assert params.solver_options == {}

    def test_full_construction(self) -> None:
        """UCParameters with all fields populated is valid."""
        gen = make_generator(startup_cost=5000.0, fuel_cost_per_mwh=25.0)
        th = TimeHorizon(num_periods=24)
        dp = DemandProfile(demands=[100.0] * 24)

        params = UCParameters(
            generators=[gen],
            demand=dp,
            time_horizon=th,
            reserve_margin=0.10,
            solver_name="HiGHS",
            solver_time_limit_s=300.0,
            mip_gap=0.01,
            solver_options={"threads": 4},
        )
        assert len(params.generators) == 1
        assert params.reserve_margin == 0.10
        assert params.solver_time_limit_s == 300.0
        assert params.mip_gap == 0.01
        assert params.solver_options == {"threads": 4}

    def test_demand_length_matches_time_horizon(self) -> None:
        """Matching demand length and time horizon is accepted."""
        th = TimeHorizon(num_periods=4)
        dp = DemandProfile(demands=[100.0, 150.0, 200.0, 180.0])
        params = UCParameters(demand=dp, time_horizon=th)
        assert len(params.demand.demands) == params.time_horizon.num_periods

    def test_demand_length_mismatch_raises(self) -> None:
        """Mismatched demand length and time horizon raises ValueError."""
        th = TimeHorizon(num_periods=24)
        dp = DemandProfile(demands=[100.0] * 12)
        with pytest.raises(ValueError, match="does not match time horizon"):
            UCParameters(demand=dp, time_horizon=th)

    def test_negative_reserve_margin_raises(self) -> None:
        """Negative reserve_margin raises ValueError."""
        with pytest.raises(ValueError, match="reserve_margin must be non-negative"):
            UCParameters(reserve_margin=-0.05)

    def test_zero_reserve_margin_valid(self) -> None:
        """Zero reserve margin is valid (no reserve requirement)."""
        params = UCParameters(reserve_margin=0.0)
        assert params.reserve_margin == 0.0

    def test_negative_time_limit_raises(self) -> None:
        """Negative solver_time_limit_s raises ValueError."""
        with pytest.raises(ValueError, match="solver_time_limit_s must be positive"):
            UCParameters(solver_time_limit_s=-10.0)

    def test_zero_time_limit_raises(self) -> None:
        """Zero solver_time_limit_s raises ValueError."""
        with pytest.raises(ValueError, match="solver_time_limit_s must be positive"):
            UCParameters(solver_time_limit_s=0.0)

    def test_mip_gap_out_of_range_raises(self) -> None:
        """mip_gap > 1 raises ValueError."""
        with pytest.raises(ValueError, match="mip_gap must be between 0 and 1"):
            UCParameters(mip_gap=1.5)

    def test_negative_mip_gap_raises(self) -> None:
        """Negative mip_gap raises ValueError."""
        with pytest.raises(ValueError, match="mip_gap must be between 0 and 1"):
            UCParameters(mip_gap=-0.01)

    def test_mip_gap_boundary_values(self) -> None:
        """mip_gap at 0.0 and 1.0 are valid boundary values."""
        params_zero = UCParameters(mip_gap=0.0)
        assert params_zero.mip_gap == 0.0

        params_one = UCParameters(mip_gap=1.0)
        assert params_one.mip_gap == 1.0

    def test_demand_only_without_time_horizon(self) -> None:
        """Demand without time horizon is valid (no cross-check)."""
        dp = DemandProfile(demands=[100.0, 200.0])
        params = UCParameters(demand=dp)
        assert params.demand is not None
        assert params.time_horizon is None

    def test_time_horizon_only_without_demand(self) -> None:
        """Time horizon without demand is valid (no cross-check)."""
        th = TimeHorizon(num_periods=24)
        params = UCParameters(time_horizon=th)
        assert params.time_horizon is not None
        assert params.demand is None

    def test_multiple_generators(self) -> None:
        """UCParameters accepts multiple generators."""
        gens = [
            make_generator(id=f"gen_{i}", name=f"Plant{i}")
            for i in range(5)
        ]
        params = UCParameters(generators=gens)
        assert len(params.generators) == 5


# ======================================================================
# GeneratorSchedule
# ======================================================================


class TestGeneratorSchedule:
    """Tests for GeneratorSchedule dataclass."""

    def test_default_construction(self) -> None:
        """GeneratorSchedule defaults are empty/zero."""
        sched = GeneratorSchedule()
        assert sched.generator_id == ""
        assert sched.commitment == []
        assert sched.power_output_mw == []
        assert sched.startup_cost == 0.0
        assert sched.shutdown_cost == 0.0
        assert sched.fuel_cost == 0.0
        assert sched.no_load_cost == 0.0

    def test_total_cost(self) -> None:
        """total_cost sums all cost components."""
        sched = GeneratorSchedule(
            generator_id="gen_001",
            startup_cost=5000.0,
            shutdown_cost=1000.0,
            fuel_cost=20000.0,
            no_load_cost=3000.0,
        )
        assert sched.total_cost == 29000.0

    def test_total_cost_zero(self) -> None:
        """total_cost is zero for default schedule."""
        sched = GeneratorSchedule()
        assert sched.total_cost == 0.0

    def test_total_energy_mwh(self) -> None:
        """total_energy_mwh sums power output across periods."""
        sched = GeneratorSchedule(
            generator_id="gen_001",
            power_output_mw=[100.0, 150.0, 200.0, 0.0],
        )
        assert sched.total_energy_mwh == 450.0

    def test_total_energy_empty(self) -> None:
        """total_energy_mwh is zero for empty power output."""
        sched = GeneratorSchedule()
        assert sched.total_energy_mwh == 0.0

    def test_num_startups_single_run(self) -> None:
        """Single continuous run has 1 startup."""
        sched = GeneratorSchedule(
            generator_id="gen_001",
            commitment=[1, 1, 1, 1],
        )
        assert sched.num_startups == 1

    def test_num_startups_multiple_runs(self) -> None:
        """Multiple on/off cycles count multiple startups."""
        sched = GeneratorSchedule(
            generator_id="gen_001",
            commitment=[1, 1, 0, 0, 1, 1, 0, 1],
        )
        # Startups: period 0 (first on), period 4 (0→1), period 7 (0→1)
        assert sched.num_startups == 3

    def test_num_startups_always_off(self) -> None:
        """All-off commitment has 0 startups."""
        sched = GeneratorSchedule(
            generator_id="gen_001",
            commitment=[0, 0, 0, 0],
        )
        assert sched.num_startups == 0

    def test_num_startups_empty(self) -> None:
        """Empty commitment has 0 startups."""
        sched = GeneratorSchedule()
        assert sched.num_startups == 0

    def test_capacity_factor_full_load(self) -> None:
        """Capacity factor is output/committed_periods for full load."""
        sched = GeneratorSchedule(
            generator_id="gen_001",
            commitment=[1, 1, 1, 1],
            power_output_mw=[100.0, 100.0, 100.0, 100.0],
        )
        assert sched.capacity_factor == 100.0

    def test_capacity_factor_partial_load(self) -> None:
        """Capacity factor reflects average output per committed period."""
        sched = GeneratorSchedule(
            generator_id="gen_001",
            commitment=[1, 1, 0, 0],
            power_output_mw=[80.0, 120.0, 0.0, 0.0],
        )
        # Average over 2 committed periods: (80 + 120) / 2 = 100.0
        # But total_output / committed_periods = 200 / 2 = 100.0
        assert sched.capacity_factor == 100.0

    def test_capacity_factor_no_commitment(self) -> None:
        """Capacity factor is 0.0 when no periods are committed."""
        sched = GeneratorSchedule(
            generator_id="gen_001",
            commitment=[0, 0, 0],
            power_output_mw=[0.0, 0.0, 0.0],
        )
        assert sched.capacity_factor == 0.0

    def test_capacity_factor_empty(self) -> None:
        """Capacity factor is 0.0 for empty schedule."""
        sched = GeneratorSchedule()
        assert sched.capacity_factor == 0.0


# ======================================================================
# UCResult
# ======================================================================


class TestUCResult:
    """Tests for UCResult dataclass and summary property."""

    def test_default_construction(self) -> None:
        """UCResult defaults to 'Not Solved' with empty data."""
        result = UCResult()
        assert result.status == "Not Solved"
        assert result.schedules == []
        assert result.total_cost == 0.0
        assert result.solve_time_s == 0.0
        assert result.gap is None
        assert result.warnings == []

    def test_optimal_result(self) -> None:
        """UCResult with Optimal status is correctly identified."""
        sched = GeneratorSchedule(
            generator_id="gen_001",
            commitment=[1, 1, 1],
            power_output_mw=[100.0, 150.0, 120.0],
            fuel_cost=9000.0,
            startup_cost=5000.0,
        )
        result = UCResult(
            status="Optimal",
            schedules=[sched],
            total_cost=14000.0,
            solve_time_s=1.23,
            gap=0.0,
        )
        assert result.is_optimal is True
        assert result.num_generators == 1
        assert result.total_cost == 14000.0

    def test_is_optimal_false(self) -> None:
        """is_optimal returns False for non-optimal statuses."""
        assert UCResult(status="Infeasible").is_optimal is False
        assert UCResult(status="Not Solved").is_optimal is False
        assert UCResult(status="Unbounded").is_optimal is False

    def test_num_generators(self) -> None:
        """num_generators counts schedules correctly."""
        scheds = [
            GeneratorSchedule(generator_id=f"gen_{i}")
            for i in range(3)
        ]
        result = UCResult(schedules=scheds)
        assert result.num_generators == 3

    def test_warnings_list(self) -> None:
        """Warnings are stored and accessible."""
        result = UCResult(
            status="Optimal",
            warnings=["Low reserve margin", "Solver near time limit"],
        )
        assert len(result.warnings) == 2
        assert "Low reserve margin" in result.warnings

    def test_summary_property(self) -> None:
        """summary returns a compact dict for logging."""
        result = UCResult(
            status="Optimal",
            total_cost=123456.789,
            solve_time_s=45.678,
            gap=0.001234,
            schedules=[
                GeneratorSchedule(generator_id="gen_001"),
                GeneratorSchedule(generator_id="gen_002"),
            ],
            warnings=["test warning"],
        )
        summary = result.summary
        assert summary["status"] == "Optimal"
        assert summary["total_cost"] == 123456.79
        assert summary["num_generators"] == 2
        assert summary["solve_time_s"] == 45.68
        assert summary["gap"] == 0.001234
        assert summary["warnings"] == 1

    def test_summary_with_none_gap(self) -> None:
        """summary handles None gap correctly."""
        result = UCResult(status="Not Solved")
        summary = result.summary
        assert summary["gap"] is None

    def test_summary_with_zero_gap(self) -> None:
        """summary handles zero gap (proven optimal)."""
        result = UCResult(status="Optimal", gap=0.0)
        summary = result.summary
        assert summary["gap"] == 0.0

    def test_empty_result_summary(self) -> None:
        """Default result has clean summary with zero values."""
        result = UCResult()
        summary = result.summary
        assert summary == {
            "status": "Not Solved",
            "total_cost": 0.0,
            "num_generators": 0,
            "solve_time_s": 0.0,
            "gap": None,
            "warnings": 0,
        }
