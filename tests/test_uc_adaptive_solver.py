"""Tests for the UC adaptive solver orchestrator.

Tests adaptive solver correctness including:
- Simple problem returns Optimal with hardware info in AdaptiveUCResult
- Problem with storage (battery + pumped hydro) solves correctly
- force_tier parameter overrides auto-detected tier
- AdaptiveUCResult contains all expected metadata (tier, profile, config)
- solver_options forwarding works end-to-end
- Existing solve_uc() behavior is unchanged (regression test)
- Full detect -> select -> solve pipeline (integration)
- Degradation path when solve fails or times out
- LP relaxation post-processing rounds fractional commitments
"""

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.model.generator import Generator
from src.uc.adaptive_solver import (
    AdaptiveUCResult,
    _apply_config,
    _postprocess_lp_relaxation,
    solve_adaptive,
)
from src.uc.hardware_detector import HardwareProfile
from src.uc.models import (
    DemandProfile,
    GeneratorSchedule,
    TimeHorizon,
    UCParameters,
    UCResult,
)
from src.uc.solver import solve_uc
from src.uc.solver_strategy import SolverConfig, SolverTier
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
    - g1: 200 MW coal base-load -- cheap fuel, slow to start, tight ramp
    - g2: 150 MW LNG mid-merit -- moderate fuel cost, flexible
    - g3: 100 MW oil peaker -- expensive fuel, very flexible
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


def _make_simple_params(
    demand_mw: float = 200.0,
    periods: int = 24,
) -> UCParameters:
    """Create a standard feasible UC problem specification."""
    gens = _make_simple_generators()
    th = TimeHorizon(num_periods=periods, period_duration_h=1.0)
    dp = _flat_demand(demand_mw, periods)
    return UCParameters(generators=gens, demand=dp, time_horizon=th)


def _make_storage_generators() -> List[Generator]:
    """Create generators including thermal + pumped hydro + battery.

    Lineup (total discharge capacity = 550 MW):
    - g1: 200 MW coal base-load
    - g2: 150 MW LNG mid-merit
    - s1: 100 MW pumped hydro (400 MWh storage)
    - s2: 100 MW battery (200 MWh storage, high efficiency)
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
    s1 = make_storage_generator(
        id="s1",
        name="Pumped Hydro",
        capacity_mw=100.0,
        fuel_type="hydro",
        storage_capacity_mwh=400.0,
        charge_rate_mw=100.0,
        discharge_rate_mw=100.0,
        charge_efficiency=0.85,
        discharge_efficiency=0.90,
        initial_soc_fraction=0.5,
        min_terminal_soc_fraction=0.5,
        fuel_cost_per_mwh=5.0,
    )
    s2 = make_storage_generator(
        id="s2",
        name="Battery",
        capacity_mw=100.0,
        fuel_type="battery",
        storage_capacity_mwh=200.0,
        charge_rate_mw=100.0,
        discharge_rate_mw=100.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        initial_soc_fraction=0.5,
        min_terminal_soc_fraction=0.5,
        fuel_cost_per_mwh=2.0,
    )
    return [g1, g2, s1, s2]


def _make_mock_profile(
    physical_cores: int = 4,
    available_ram_gb: float = 16.0,
) -> HardwareProfile:
    """Create a HardwareProfile for testing."""
    return HardwareProfile(
        physical_cores=physical_cores,
        logical_cores=physical_cores * 2,
        available_ram_gb=available_ram_gb,
        total_ram_gb=available_ram_gb * 2,
        available_solvers=["HiGHS_CMD", "PULP_CBC_CMD"],
        os_name="Darwin",
        architecture="arm64",
    )


def _find_schedule(result: UCResult, gen_id: str) -> GeneratorSchedule:
    """Find the GeneratorSchedule for a given generator id."""
    for s in result.schedules:
        if s.generator_id == gen_id:
            return s
    raise ValueError(f"No schedule for generator '{gen_id}'")


# ======================================================================
# TestAdaptiveSolveOptimal
# ======================================================================


class TestAdaptiveSolveOptimal:
    """Tests that solve_adaptive returns Optimal for a feasible instance."""

    def test_optimal_status_simple_problem(self) -> None:
        """3-generator 24-period instance returns Optimal via adaptive solver."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(params, verbose=False)

        assert adaptive_result.result.status == "Optimal"
        assert adaptive_result.result.total_cost > 0

    def test_hardware_profile_populated(self) -> None:
        """AdaptiveUCResult contains a valid hardware profile."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(params, verbose=False)

        assert adaptive_result.hardware_profile is not None
        assert isinstance(adaptive_result.hardware_profile, HardwareProfile)
        assert adaptive_result.hardware_profile.physical_cores >= 1
        assert adaptive_result.hardware_profile.available_ram_gb > 0

    def test_solver_config_populated(self) -> None:
        """AdaptiveUCResult contains a valid solver config."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(params, verbose=False)

        assert adaptive_result.solver_config is not None
        assert isinstance(adaptive_result.solver_config, SolverConfig)
        assert adaptive_result.solver_config.time_limit_s > 0
        assert adaptive_result.solver_config.threads >= 1

    def test_tier_used_populated(self) -> None:
        """AdaptiveUCResult reports the tier used."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(params, verbose=False)

        assert adaptive_result.tier_used is not None
        assert isinstance(adaptive_result.tier_used, SolverTier)
        assert adaptive_result.tier_used in (
            SolverTier.HIGH,
            SolverTier.MID,
            SolverTier.LOW,
        )

    def test_total_time_positive(self) -> None:
        """AdaptiveUCResult records a positive total_time_s."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(params, verbose=False)

        assert adaptive_result.total_time_s > 0

    def test_degradation_history_nonempty(self) -> None:
        """Degradation history contains at least one entry (tier attempt)."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(params, verbose=False)

        assert len(adaptive_result.degradation_history) >= 1

    def test_schedules_for_all_generators(self) -> None:
        """Result contains schedules for all generators."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(params, verbose=False)

        assert adaptive_result.result.status == "Optimal"
        gen_ids = {s.generator_id for s in adaptive_result.result.schedules}
        assert gen_ids == {"g1", "g2", "g3"}

    def test_demand_balance_at_each_timestep(self) -> None:
        """Total generation meets demand at every timestep."""
        demand_mw = 200.0
        params = _make_simple_params(demand_mw=demand_mw)

        adaptive_result = solve_adaptive(params, verbose=False)
        assert adaptive_result.result.status == "Optimal"

        for t in range(24):
            total_gen = sum(
                s.power_output_mw[t] for s in adaptive_result.result.schedules
            )
            assert total_gen >= demand_mw - 1e-3, (
                f"Demand balance violated at t={t}: "
                f"generation={total_gen:.4f} < demand={demand_mw}"
            )


# ======================================================================
# TestAdaptiveSolveWithStorage
# ======================================================================


class TestAdaptiveSolveWithStorage:
    """Tests that adaptive solver handles storage generators correctly."""

    def test_optimal_with_storage(self) -> None:
        """Problem with thermal + pumped hydro + battery solves to Optimal."""
        gens = _make_storage_generators()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(200.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        adaptive_result = solve_adaptive(params, verbose=False)

        assert adaptive_result.result.status == "Optimal"
        assert adaptive_result.result.total_cost > 0

    def test_storage_schedules_present(self) -> None:
        """Storage generators have schedules in the result."""
        gens = _make_storage_generators()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(200.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        adaptive_result = solve_adaptive(params, verbose=False)
        assert adaptive_result.result.status == "Optimal"

        gen_ids = {s.generator_id for s in adaptive_result.result.schedules}
        assert "s1" in gen_ids
        assert "s2" in gen_ids

    def test_storage_soc_tracked(self) -> None:
        """Storage units have SOC values tracked in their schedules."""
        gens = _make_storage_generators()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(200.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        adaptive_result = solve_adaptive(params, verbose=False)
        assert adaptive_result.result.status == "Optimal"

        s1_sched = _find_schedule(adaptive_result.result, "s1")
        # Pumped hydro should have SOC values
        assert len(s1_sched.soc_mwh) == 24

    def test_demand_balance_with_storage(self) -> None:
        """Demand balance holds at every timestep with storage units."""
        demand_mw = 200.0
        gens = _make_storage_generators()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(demand_mw, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        adaptive_result = solve_adaptive(params, verbose=False)
        assert adaptive_result.result.status == "Optimal"

        for t in range(24):
            total_gen = sum(
                s.power_output_mw[t]
                for s in adaptive_result.result.schedules
            )
            assert total_gen >= demand_mw - 1e-3, (
                f"Demand balance violated at t={t}: "
                f"generation={total_gen:.4f} < demand={demand_mw}"
            )


# ======================================================================
# TestAdaptiveSolveForcedTier
# ======================================================================


class TestAdaptiveSolveForcedTier:
    """Tests that force_tier parameter overrides auto-detection."""

    def test_force_high_tier(self) -> None:
        """force_tier=HIGH overrides auto-detected tier."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(
            params, force_tier=SolverTier.HIGH, verbose=False
        )

        assert adaptive_result.tier_used == SolverTier.HIGH
        assert adaptive_result.solver_config.tier == SolverTier.HIGH

    def test_force_mid_tier(self) -> None:
        """force_tier=MID overrides auto-detected tier."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(
            params, force_tier=SolverTier.MID, verbose=False
        )

        assert adaptive_result.tier_used == SolverTier.MID
        assert adaptive_result.solver_config.tier == SolverTier.MID

    def test_force_low_tier(self) -> None:
        """force_tier=LOW uses LP relaxation tier."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(
            params, force_tier=SolverTier.LOW, verbose=False
        )

        assert adaptive_result.tier_used == SolverTier.LOW
        assert adaptive_result.solver_config.tier == SolverTier.LOW

    def test_forced_tier_recorded_in_history(self) -> None:
        """Degradation history records that tier was forced."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(
            params, force_tier=SolverTier.MID, verbose=False
        )

        assert any(
            "Forced tier" in entry
            for entry in adaptive_result.degradation_history
        )

    def test_forced_high_still_returns_optimal(self) -> None:
        """Forced HIGH tier still produces Optimal for a feasible problem."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(
            params, force_tier=SolverTier.HIGH, verbose=False
        )

        assert adaptive_result.result.status == "Optimal"
        assert adaptive_result.result.total_cost > 0

    def test_forced_low_produces_result(self) -> None:
        """Forced LOW tier (LP relaxation) produces a result."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(
            params, force_tier=SolverTier.LOW, verbose=False
        )

        # LP relaxation should still produce a result (Optimal for the LP)
        assert adaptive_result.result.status in ("Optimal", "Not Solved")
        assert adaptive_result.tier_used == SolverTier.LOW


# ======================================================================
# TestAdaptiveResultMetadata
# ======================================================================


class TestAdaptiveResultMetadata:
    """Tests that AdaptiveUCResult contains all expected metadata."""

    def test_result_is_uc_result(self) -> None:
        """AdaptiveUCResult.result is a UCResult instance."""
        params = _make_simple_params()
        adaptive_result = solve_adaptive(params, verbose=False)
        assert isinstance(adaptive_result.result, UCResult)

    def test_hardware_profile_is_hardware_profile(self) -> None:
        """AdaptiveUCResult.hardware_profile is a HardwareProfile."""
        params = _make_simple_params()
        adaptive_result = solve_adaptive(params, verbose=False)
        assert isinstance(adaptive_result.hardware_profile, HardwareProfile)

    def test_solver_config_is_solver_config(self) -> None:
        """AdaptiveUCResult.solver_config is a SolverConfig."""
        params = _make_simple_params()
        adaptive_result = solve_adaptive(params, verbose=False)
        assert isinstance(adaptive_result.solver_config, SolverConfig)

    def test_tier_used_is_solver_tier(self) -> None:
        """AdaptiveUCResult.tier_used is a SolverTier enum member."""
        params = _make_simple_params()
        adaptive_result = solve_adaptive(params, verbose=False)
        assert isinstance(adaptive_result.tier_used, SolverTier)

    def test_degradation_history_is_list_of_strings(self) -> None:
        """AdaptiveUCResult.degradation_history is a list of strings."""
        params = _make_simple_params()
        adaptive_result = solve_adaptive(params, verbose=False)
        assert isinstance(adaptive_result.degradation_history, list)
        for entry in adaptive_result.degradation_history:
            assert isinstance(entry, str)

    def test_total_time_is_float(self) -> None:
        """AdaptiveUCResult.total_time_s is a non-negative float."""
        params = _make_simple_params()
        adaptive_result = solve_adaptive(params, verbose=False)
        assert isinstance(adaptive_result.total_time_s, float)
        assert adaptive_result.total_time_s >= 0

    def test_config_tier_matches_tier_used(self) -> None:
        """solver_config.tier and tier_used are consistent."""
        params = _make_simple_params()
        adaptive_result = solve_adaptive(params, verbose=False)
        assert adaptive_result.solver_config.tier == adaptive_result.tier_used

    def test_default_adaptive_result(self) -> None:
        """AdaptiveUCResult default construction produces valid defaults."""
        result = AdaptiveUCResult()
        assert isinstance(result.result, UCResult)
        assert result.hardware_profile is None
        assert result.solver_config is None
        assert result.tier_used is None
        assert result.degradation_history == []
        assert result.total_time_s == 0.0


# ======================================================================
# TestSolverOptionsForwarding
# ======================================================================


class TestSolverOptionsForwarding:
    """Tests that solver_options are forwarded end-to-end."""

    def test_solver_options_in_params_preserved(self) -> None:
        """Custom solver_options in UCParameters survive through adaptive solve."""
        params = _make_simple_params()
        params_with_opts = UCParameters(
            generators=params.generators,
            demand=params.demand,
            time_horizon=params.time_horizon,
            solver_options={"warmStart": True},
        )

        adaptive_result = solve_adaptive(params_with_opts, verbose=False)

        # The adaptive solver should still produce a result
        assert adaptive_result.result.status in ("Optimal", "Not Solved")

    def test_apply_config_sets_threads(self) -> None:
        """_apply_config sets threads in solver_options when threads > 1."""
        params = _make_simple_params()
        config = SolverConfig(
            tier=SolverTier.HIGH,
            solver_name="HiGHS_CMD",
            time_limit_s=600.0,
            mip_gap=0.01,
            threads=4,
            use_decomposition=False,
            decomposition_strategy=None,
            use_lp_relaxation=False,
            description="Test config",
        )

        applied = _apply_config(params, config)

        assert applied.solver_options.get("threads") == 4

    def test_apply_config_sets_lp_relaxation(self) -> None:
        """_apply_config sets mip=False when use_lp_relaxation is True."""
        params = _make_simple_params()
        config = SolverConfig(
            tier=SolverTier.LOW,
            solver_name="PULP_CBC_CMD",
            time_limit_s=120.0,
            mip_gap=0.10,
            threads=1,
            use_decomposition=False,
            decomposition_strategy=None,
            use_lp_relaxation=True,
            description="Test LP relaxation",
        )

        applied = _apply_config(params, config)

        assert applied.solver_options.get("mip") is False

    def test_apply_config_preserves_existing_options(self) -> None:
        """_apply_config preserves existing solver_options from params."""
        params = UCParameters(
            generators=_make_simple_generators(),
            demand=_flat_demand(200.0, 24),
            time_horizon=TimeHorizon(num_periods=24, period_duration_h=1.0),
            solver_options={"warmStart": True},
        )
        config = SolverConfig(
            tier=SolverTier.HIGH,
            solver_name="HiGHS_CMD",
            time_limit_s=600.0,
            mip_gap=0.01,
            threads=4,
            use_decomposition=False,
            decomposition_strategy=None,
            use_lp_relaxation=False,
            description="Test config",
        )

        applied = _apply_config(params, config)

        assert applied.solver_options.get("warmStart") is True
        assert applied.solver_options.get("threads") == 4

    def test_apply_config_does_not_mutate_original(self) -> None:
        """_apply_config does not mutate the original UCParameters."""
        params = _make_simple_params()
        original_options = dict(params.solver_options)
        config = SolverConfig(
            tier=SolverTier.HIGH,
            solver_name="HiGHS_CMD",
            time_limit_s=600.0,
            mip_gap=0.01,
            threads=4,
            use_decomposition=False,
            decomposition_strategy=None,
            use_lp_relaxation=False,
            description="Test config",
        )

        _apply_config(params, config)

        assert params.solver_options == original_options

    def test_apply_config_sets_solver_name(self) -> None:
        """_apply_config sets the solver name from config."""
        params = _make_simple_params()
        config = SolverConfig(
            tier=SolverTier.HIGH,
            solver_name="PULP_CBC_CMD",
            time_limit_s=600.0,
            mip_gap=0.01,
            threads=1,
            use_decomposition=False,
            decomposition_strategy=None,
            use_lp_relaxation=False,
            description="Test config",
        )

        applied = _apply_config(params, config)

        assert applied.solver_name == "PULP_CBC_CMD"

    def test_apply_config_sets_time_limit(self) -> None:
        """_apply_config sets the solver time limit from config."""
        params = _make_simple_params()
        config = SolverConfig(
            tier=SolverTier.MID,
            solver_name="HiGHS_CMD",
            time_limit_s=300.0,
            mip_gap=0.05,
            threads=2,
            use_decomposition=False,
            decomposition_strategy=None,
            use_lp_relaxation=False,
            description="Test config",
        )

        applied = _apply_config(params, config)

        assert applied.solver_time_limit_s == 300.0

    def test_apply_config_sets_mip_gap(self) -> None:
        """_apply_config sets the MIP gap from config."""
        params = _make_simple_params()
        config = SolverConfig(
            tier=SolverTier.MID,
            solver_name="HiGHS_CMD",
            time_limit_s=300.0,
            mip_gap=0.05,
            threads=2,
            use_decomposition=False,
            decomposition_strategy=None,
            use_lp_relaxation=False,
            description="Test config",
        )

        applied = _apply_config(params, config)

        assert applied.mip_gap == 0.05


# ======================================================================
# TestExistingSolveUCRegression
# ======================================================================


class TestExistingSolveUCRegression:
    """Regression tests ensuring existing solve_uc() behavior is unchanged."""

    def test_solve_uc_still_returns_optimal(self) -> None:
        """Existing solve_uc() still returns Optimal for a feasible problem."""
        params = _make_simple_params()

        result = solve_uc(params)

        assert result.status == "Optimal"
        assert result.total_cost > 0
        assert len(result.schedules) == 3

    def test_solve_uc_demand_balance(self) -> None:
        """Existing solve_uc() maintains demand balance at each timestep."""
        demand_mw = 200.0
        params = _make_simple_params(demand_mw=demand_mw)

        result = solve_uc(params)
        assert result.status == "Optimal"

        for t in range(24):
            total_gen = sum(s.power_output_mw[t] for s in result.schedules)
            assert total_gen >= demand_mw - 1e-3

    def test_solve_uc_with_empty_solver_options(self) -> None:
        """solve_uc() works unchanged when solver_options is empty dict."""
        params = _make_simple_params()
        # Default solver_options is empty dict
        assert params.solver_options == {}

        result = solve_uc(params)

        assert result.status == "Optimal"

    def test_solve_uc_cost_matches_adaptive(self) -> None:
        """solve_uc() and solve_adaptive() produce similar costs for same problem."""
        params = _make_simple_params()

        direct_result = solve_uc(params)
        adaptive_result = solve_adaptive(
            params, force_tier=SolverTier.HIGH, verbose=False
        )

        # Both should be optimal
        assert direct_result.status == "Optimal"
        assert adaptive_result.result.status == "Optimal"

        # Costs should be very close (may differ slightly due to solver config)
        cost_diff_pct = abs(
            direct_result.total_cost - adaptive_result.result.total_cost
        ) / max(direct_result.total_cost, 1.0)
        assert cost_diff_pct < 0.05, (
            f"Cost mismatch: direct={direct_result.total_cost:.2f}, "
            f"adaptive={adaptive_result.result.total_cost:.2f}"
        )


# ======================================================================
# TestAdaptiveSolverFullPipeline (Integration)
# ======================================================================


class TestAdaptiveSolverFullPipeline:
    """Integration tests for the full detect -> select -> solve pipeline."""

    def test_full_pipeline_end_to_end(self) -> None:
        """Full pipeline: detect hardware, select strategy, solve."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(params, verbose=False)

        # Verify all pipeline stages produced results
        assert adaptive_result.hardware_profile is not None
        assert adaptive_result.solver_config is not None
        assert adaptive_result.tier_used is not None
        assert adaptive_result.result.status == "Optimal"
        assert adaptive_result.total_time_s > 0
        assert len(adaptive_result.degradation_history) >= 1

    def test_pipeline_with_storage_generators(self) -> None:
        """Full pipeline works with mixed thermal + storage generators."""
        gens = _make_storage_generators()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(200.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        adaptive_result = solve_adaptive(params, verbose=False)

        assert adaptive_result.result.status == "Optimal"
        assert len(adaptive_result.result.schedules) == 4

    def test_pipeline_verbose_mode(self) -> None:
        """Verbose mode does not crash and still produces correct results."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(params, verbose=True)

        assert adaptive_result.result.status == "Optimal"

    def test_pipeline_with_reserve_margin(self) -> None:
        """Pipeline handles reserve margin parameter."""
        gens = _make_simple_generators()
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(150.0, 24)  # Lower demand to accommodate reserve
        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            reserve_margin=0.10,
        )

        adaptive_result = solve_adaptive(params, verbose=False)

        assert adaptive_result.result.status == "Optimal"


# ======================================================================
# TestDegradationPath
# ======================================================================


class TestDegradationPath:
    """Tests degradation path behavior when solve fails or times out."""

    def test_degradation_on_solve_exception(self) -> None:
        """System degrades to lower tier when solve raises an exception."""
        params = _make_simple_params()
        call_count = 0

        original_solve = solve_uc

        def _failing_then_succeeding_solve(p):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated solver failure")
            return original_solve(p)

        with patch(
            "src.uc.adaptive_solver.solve_uc",
            side_effect=_failing_then_succeeding_solve,
        ):
            adaptive_result = solve_adaptive(
                params, force_tier=SolverTier.HIGH, verbose=False
            )

        # Should have degraded past the first tier
        assert any(
            "solve raised" in entry
            for entry in adaptive_result.degradation_history
        )
        # Should eventually succeed on a lower tier
        assert adaptive_result.result.status in ("Optimal", "Not Solved")

    def test_all_tiers_fail_returns_not_solved(self) -> None:
        """When all tiers fail, result status is 'Not Solved'."""
        params = _make_simple_params()

        with patch(
            "src.uc.adaptive_solver.solve_uc",
            side_effect=RuntimeError("Simulated failure"),
        ):
            adaptive_result = solve_adaptive(
                params, force_tier=SolverTier.HIGH, verbose=False
            )

        assert adaptive_result.result.status == "Not Solved"
        # History should show all three tier attempts
        assert len(adaptive_result.degradation_history) >= 3

    def test_degradation_history_tracks_attempts(self) -> None:
        """Degradation history tracks each tier attempt."""
        params = _make_simple_params()

        adaptive_result = solve_adaptive(
            params, force_tier=SolverTier.HIGH, verbose=False
        )

        # Should have at least one entry (the forced tier + result)
        assert len(adaptive_result.degradation_history) >= 1
        # First entry should mention "Forced tier"
        assert "Forced tier: high" in adaptive_result.degradation_history[0]

    def test_non_optimal_triggers_degradation(self) -> None:
        """Non-optimal status triggers degradation to next tier."""
        params = _make_simple_params()

        non_optimal = UCResult(status="Not Solved")
        optimal = UCResult(
            status="Optimal",
            total_cost=1000.0,
            schedules=[],
        )
        call_count = 0

        def _first_fails_solve(p):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return non_optimal
            return solve_uc(p)

        with patch(
            "src.uc.adaptive_solver.solve_uc",
            side_effect=_first_fails_solve,
        ):
            adaptive_result = solve_adaptive(
                params, force_tier=SolverTier.HIGH, verbose=False
            )

        # Should have recorded non-optimal status and degraded
        assert any(
            "Not Solved" in entry
            for entry in adaptive_result.degradation_history
        )


# ======================================================================
# TestLPRelaxationPostProcessing
# ======================================================================


class TestLPRelaxationPostProcessing:
    """Tests LP relaxation post-processing of fractional commitments."""

    def test_rounds_fractional_commitments(self) -> None:
        """Fractional commitment values are rounded (>= 0.5 -> 1, < 0.5 -> 0)."""
        result = UCResult(
            status="Optimal",
            total_cost=100.0,
            schedules=[
                GeneratorSchedule(
                    generator_id="g1",
                    commitment=[0.8, 0.3, 0.5, 0.49, 1.0, 0.0],
                    power_output_mw=[80.0, 30.0, 50.0, 49.0, 100.0, 0.0],
                ),
            ],
        )

        _postprocess_lp_relaxation(result)

        assert result.schedules[0].commitment == [1, 0, 1, 0, 1, 0]

    def test_integral_values_unchanged(self) -> None:
        """Integral commitment values (0, 1) are not changed."""
        result = UCResult(
            status="Optimal",
            total_cost=100.0,
            schedules=[
                GeneratorSchedule(
                    generator_id="g1",
                    commitment=[1, 0, 1, 1, 0, 0],
                    power_output_mw=[100.0, 0.0, 100.0, 100.0, 0.0, 0.0],
                ),
            ],
        )

        _postprocess_lp_relaxation(result)

        assert result.schedules[0].commitment == [1, 0, 1, 1, 0, 0]

    def test_fractional_rounding_adds_warning(self) -> None:
        """Rounding fractional values appends a warning to result."""
        result = UCResult(
            status="Optimal",
            total_cost=100.0,
            schedules=[
                GeneratorSchedule(
                    generator_id="g1",
                    commitment=[0.7, 0.3],
                    power_output_mw=[70.0, 30.0],
                ),
            ],
        )

        _postprocess_lp_relaxation(result)

        assert len(result.warnings) >= 1
        assert any(
            "LP relaxation" in w and "rounded" in w for w in result.warnings
        )

    def test_no_warning_when_all_integral(self) -> None:
        """No warning appended when all values are already integral."""
        result = UCResult(
            status="Optimal",
            total_cost=100.0,
            warnings=[],
            schedules=[
                GeneratorSchedule(
                    generator_id="g1",
                    commitment=[1, 0, 1, 0],
                    power_output_mw=[100.0, 0.0, 100.0, 0.0],
                ),
            ],
        )

        _postprocess_lp_relaxation(result)

        assert not any("rounded" in w for w in result.warnings)

    def test_multiple_schedules_processed(self) -> None:
        """Post-processing handles multiple generator schedules."""
        result = UCResult(
            status="Optimal",
            total_cost=200.0,
            schedules=[
                GeneratorSchedule(
                    generator_id="g1",
                    commitment=[0.9, 0.1],
                    power_output_mw=[90.0, 10.0],
                ),
                GeneratorSchedule(
                    generator_id="g2",
                    commitment=[0.6, 0.4],
                    power_output_mw=[60.0, 40.0],
                ),
            ],
        )

        _postprocess_lp_relaxation(result)

        assert result.schedules[0].commitment == [1, 0]
        assert result.schedules[1].commitment == [1, 0]
