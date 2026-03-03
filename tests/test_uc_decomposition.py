"""Comprehensive tests for the UC decomposition module.

Tests decomposition correctness including:
- RegionalDecomposer partitions generators correctly by region
- FuelTypeDecomposer partitions generators by fuel type
- TimeWindowDecomposer splits time horizon into overlapping windows
- Empty partition handling (region with no generators)
- Decomposed solve produces feasible combined schedule
- All generators accounted for (no missing/duplicated)
- Result merging produces correct total cost
- Consistency at time-window boundaries
- Factory function create_decomposer
"""

from typing import Dict, List, Set

import pytest

from src.model.generator import Generator
from src.uc.decomposition import (
    Decomposer,
    FuelTypeDecomposer,
    RegionalDecomposer,
    TimeWindowDecomposer,
    create_decomposer,
)
from src.uc.models import DemandProfile, TimeHorizon, UCParameters, UCResult
from src.uc.solver import solve_uc
from tests.conftest import make_generator


# ======================================================================
# Helpers
# ======================================================================


def _flat_demand(mw: float, periods: int) -> DemandProfile:
    """Create a constant demand profile."""
    return DemandProfile(demands=[mw] * periods)


def _make_regional_generators() -> List[Generator]:
    """Create generators across two regions for regional decomposition tests.

    Returns 4 generators in 2 regions (total 500 MW):
    - shikoku: g_shikoku_1 (200 MW coal), g_shikoku_2 (100 MW lng)
    - hokkaido: g_hokkaido_1 (150 MW coal), g_hokkaido_2 (50 MW oil)
    """
    g1 = make_generator(
        id="g_shikoku_1",
        name="Shikoku Coal",
        capacity_mw=200.0,
        fuel_type="coal",
        region="shikoku",
        p_min_mw=50.0,
        fuel_cost_per_mwh=30.0,
        no_load_cost=100.0,
        startup_cost=5000.0,
    )
    g2 = make_generator(
        id="g_shikoku_2",
        name="Shikoku LNG",
        capacity_mw=100.0,
        fuel_type="lng",
        region="shikoku",
        p_min_mw=20.0,
        fuel_cost_per_mwh=50.0,
        no_load_cost=50.0,
        startup_cost=2000.0,
    )
    g3 = make_generator(
        id="g_hokkaido_1",
        name="Hokkaido Coal",
        capacity_mw=150.0,
        fuel_type="coal",
        region="hokkaido",
        p_min_mw=30.0,
        fuel_cost_per_mwh=35.0,
        no_load_cost=80.0,
        startup_cost=4000.0,
    )
    g4 = make_generator(
        id="g_hokkaido_2",
        name="Hokkaido Oil",
        capacity_mw=50.0,
        fuel_type="oil",
        region="hokkaido",
        p_min_mw=10.0,
        fuel_cost_per_mwh=80.0,
        no_load_cost=20.0,
        startup_cost=1000.0,
    )
    return [g1, g2, g3, g4]


def _make_fuel_type_generators() -> List[Generator]:
    """Create generators with distinct fuel types for fuel-type decomposition.

    Returns 4 generators with 3 fuel types (total 500 MW):
    - coal: g_coal_1 (200 MW), g_coal_2 (100 MW)
    - lng: g_lng_1 (150 MW)
    - oil: g_oil_1 (50 MW)
    """
    g1 = make_generator(
        id="g_coal_1",
        name="Coal Plant 1",
        capacity_mw=200.0,
        fuel_type="coal",
        region="shikoku",
        p_min_mw=50.0,
        fuel_cost_per_mwh=30.0,
        no_load_cost=100.0,
        startup_cost=5000.0,
    )
    g2 = make_generator(
        id="g_coal_2",
        name="Coal Plant 2",
        capacity_mw=100.0,
        fuel_type="coal",
        region="shikoku",
        p_min_mw=20.0,
        fuel_cost_per_mwh=35.0,
        no_load_cost=60.0,
        startup_cost=3000.0,
    )
    g3 = make_generator(
        id="g_lng_1",
        name="LNG Plant",
        capacity_mw=150.0,
        fuel_type="lng",
        region="hokkaido",
        p_min_mw=30.0,
        fuel_cost_per_mwh=50.0,
        no_load_cost=50.0,
        startup_cost=2000.0,
    )
    g4 = make_generator(
        id="g_oil_1",
        name="Oil Peaker",
        capacity_mw=50.0,
        fuel_type="oil",
        region="hokkaido",
        p_min_mw=10.0,
        fuel_cost_per_mwh=80.0,
        no_load_cost=20.0,
        startup_cost=1000.0,
    )
    return [g1, g2, g3, g4]


def _collect_gen_ids(params_list: List[UCParameters]) -> Set[str]:
    """Collect all generator IDs across partitions."""
    ids: Set[str] = set()
    for p in params_list:
        for g in p.generators:
            ids.add(g.id)
    return ids


def _collect_gen_ids_list(params_list: List[UCParameters]) -> List[str]:
    """Collect all generator IDs across partitions (preserving duplicates)."""
    ids: List[str] = []
    for p in params_list:
        for g in p.generators:
            ids.append(g.id)
    return ids


def _find_schedule(result: UCResult, gen_id: str):
    """Find the GeneratorSchedule for a given generator id."""
    for s in result.schedules:
        if s.generator_id == gen_id:
            return s
    raise ValueError(f"No schedule for generator '{gen_id}'")


# ======================================================================
# TestRegionalDecomposer — partition by region
# ======================================================================


class TestRegionalDecomposerPartition:
    """Tests that RegionalDecomposer partitions generators by region."""

    def test_partitions_by_region(self) -> None:
        """Generators are grouped by their region attribute."""
        gens = _make_regional_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)

        assert len(partitions) == 2  # shikoku + hokkaido

        # Verify each partition has only generators from one region
        for p in partitions:
            regions = {g.region for g in p.generators}
            assert len(regions) == 1, (
                f"Partition has generators from multiple regions: {regions}"
            )

    def test_correct_generators_per_region(self) -> None:
        """Each region partition contains the correct generators."""
        gens = _make_regional_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)

        # Build region -> gen_ids map from partitions
        region_gen_ids: Dict[str, Set[str]] = {}
        for p in partitions:
            region = p.generators[0].region
            region_gen_ids[region] = {g.id for g in p.generators}

        assert region_gen_ids["shikoku"] == {"g_shikoku_1", "g_shikoku_2"}
        assert region_gen_ids["hokkaido"] == {"g_hokkaido_1", "g_hokkaido_2"}

    def test_all_generators_accounted_for(self) -> None:
        """No generators missing or duplicated across partitions."""
        gens = _make_regional_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)

        all_ids = _collect_gen_ids_list(partitions)
        original_ids = [g.id for g in gens]

        # No missing generators
        assert set(all_ids) == set(original_ids)
        # No duplicated generators
        assert len(all_ids) == len(original_ids)

    def test_demand_split_proportional_to_capacity(self) -> None:
        """Demand is split proportionally to each region's total capacity."""
        gens = _make_regional_generators()
        # shikoku: 200 + 100 = 300 MW (60%)
        # hokkaido: 150 + 50 = 200 MW (40%)
        total_cap = 500.0
        total_demand = 200.0

        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(total_demand, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)

        for p in partitions:
            region = p.generators[0].region
            region_cap = sum(g.capacity_mw for g in p.generators)
            expected_frac = region_cap / total_cap
            for t, demand_val in enumerate(p.demand.demands):
                expected = total_demand * expected_frac
                assert abs(demand_val - expected) < 1e-6, (
                    f"Region {region}, t={t}: demand={demand_val:.4f}, "
                    f"expected={expected:.4f}"
                )

    def test_single_region_returns_original(self) -> None:
        """Single-region generators return [params] unchanged."""
        gens = [
            make_generator(id="g1", name="G1", capacity_mw=200.0,
                           fuel_type="coal", region="shikoku",
                           fuel_cost_per_mwh=30.0),
            make_generator(id="g2", name="G2", capacity_mw=100.0,
                           fuel_type="lng", region="shikoku",
                           fuel_cost_per_mwh=50.0),
        ]
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)

        assert len(partitions) == 1
        assert partitions[0] is params  # should return the original object

    def test_empty_generators_returns_empty(self) -> None:
        """No generators returns empty partition list."""
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)
        params = UCParameters(generators=[], demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)

        assert partitions == []

    def test_solver_config_preserved_in_partitions(self) -> None:
        """Solver configuration is carried over to sub-problems."""
        gens = _make_regional_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)
        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            reserve_margin=0.10,
            solver_time_limit_s=30.0,
            mip_gap=0.02,
        )

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)

        for p in partitions:
            assert p.reserve_margin == 0.10
            assert p.solver_time_limit_s == 30.0
            assert p.mip_gap == 0.02


# ======================================================================
# TestFuelTypeDecomposer — partition by fuel type
# ======================================================================


class TestFuelTypeDecomposerPartition:
    """Tests that FuelTypeDecomposer partitions generators by fuel type."""

    def test_partitions_by_fuel_type(self) -> None:
        """Generators are grouped by their fuel_type_enum value."""
        gens = _make_fuel_type_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = FuelTypeDecomposer()
        partitions = decomposer.partition(params)

        assert len(partitions) == 3  # coal, lng, oil

    def test_correct_generators_per_fuel_type(self) -> None:
        """Each fuel-type partition contains the correct generators."""
        gens = _make_fuel_type_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = FuelTypeDecomposer()
        partitions = decomposer.partition(params)

        fuel_gen_ids: Dict[str, Set[str]] = {}
        for p in partitions:
            ft = p.generators[0].fuel_type_enum.value
            fuel_gen_ids[ft] = {g.id for g in p.generators}

        assert fuel_gen_ids["coal"] == {"g_coal_1", "g_coal_2"}
        assert fuel_gen_ids["lng"] == {"g_lng_1"}
        assert fuel_gen_ids["oil"] == {"g_oil_1"}

    def test_all_generators_accounted_for(self) -> None:
        """No generators missing or duplicated across fuel-type partitions."""
        gens = _make_fuel_type_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = FuelTypeDecomposer()
        partitions = decomposer.partition(params)

        all_ids = _collect_gen_ids_list(partitions)
        original_ids = [g.id for g in gens]

        assert set(all_ids) == set(original_ids)
        assert len(all_ids) == len(original_ids)

    def test_single_fuel_type_returns_original(self) -> None:
        """All generators sharing one fuel type returns [params] unchanged."""
        gens = [
            make_generator(id="g1", name="G1", capacity_mw=200.0,
                           fuel_type="coal", fuel_cost_per_mwh=30.0),
            make_generator(id="g2", name="G2", capacity_mw=100.0,
                           fuel_type="coal", fuel_cost_per_mwh=35.0),
        ]
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = FuelTypeDecomposer()
        partitions = decomposer.partition(params)

        assert len(partitions) == 1
        assert partitions[0] is params

    def test_demand_split_proportional_to_capacity(self) -> None:
        """Demand is split proportionally to each fuel group's capacity."""
        gens = _make_fuel_type_generators()
        # coal: 200 + 100 = 300 MW (60%)
        # lng: 150 MW (30%)
        # oil: 50 MW (10%)
        total_cap = 500.0
        total_demand = 250.0

        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(total_demand, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = FuelTypeDecomposer()
        partitions = decomposer.partition(params)

        for p in partitions:
            group_cap = sum(g.capacity_mw for g in p.generators)
            expected_frac = group_cap / total_cap
            for demand_val in p.demand.demands:
                expected = total_demand * expected_frac
                assert abs(demand_val - expected) < 1e-6


# ======================================================================
# TestTimeWindowDecomposer — split time horizon
# ======================================================================


class TestTimeWindowDecomposerPartition:
    """Tests that TimeWindowDecomposer splits the time horizon correctly."""

    def test_splits_horizon_into_windows(self) -> None:
        """A 24-period horizon with window_size=12, overlap=2 yields 3 windows."""
        gens = [
            make_generator(id="g1", name="G1", capacity_mw=200.0,
                           fuel_type="coal", fuel_cost_per_mwh=30.0,
                           p_min_mw=50.0, no_load_cost=10.0),
        ]
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(100.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = TimeWindowDecomposer(window_size=12, overlap=2)
        partitions = decomposer.partition(params)

        # step = 12 - 2 = 10
        # Window 0: periods 0-11 (12 periods)
        # Window 1: periods 10-21 (12 periods)
        # Window 2: periods 20-23 (4 periods)
        assert len(partitions) == 3

    def test_each_window_has_all_generators(self) -> None:
        """All generators appear in every time-window partition."""
        gens = _make_regional_generators()  # 4 generators
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(100.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = TimeWindowDecomposer(window_size=12, overlap=2)
        partitions = decomposer.partition(params)

        for p in partitions:
            gen_ids = {g.id for g in p.generators}
            original_ids = {g.id for g in gens}
            assert gen_ids == original_ids, (
                f"Window missing generators: {original_ids - gen_ids}"
            )

    def test_window_demand_slices_correct(self) -> None:
        """Each window's demand matches the correct slice of the original."""
        gens = [
            make_generator(id="g1", name="G1", capacity_mw=200.0,
                           fuel_type="coal", fuel_cost_per_mwh=30.0),
        ]
        demands = list(range(1, 25))  # [1, 2, 3, ..., 24]
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = DemandProfile(demands=[float(d) for d in demands])
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = TimeWindowDecomposer(window_size=12, overlap=2)
        partitions = decomposer.partition(params)

        # Window 0: periods 0-11 → demands[0:12]
        assert partitions[0].demand.demands == [float(d) for d in range(1, 13)]
        # Window 1: periods 10-21 → demands[10:22]
        assert partitions[1].demand.demands == [float(d) for d in range(11, 23)]
        # Window 2: periods 20-23 → demands[20:24]
        assert partitions[2].demand.demands == [float(d) for d in range(21, 25)]

    def test_small_horizon_no_decomposition(self) -> None:
        """Horizon <= window_size returns [params] without decomposition."""
        gens = [
            make_generator(id="g1", name="G1", capacity_mw=200.0,
                           fuel_type="coal", fuel_cost_per_mwh=30.0),
        ]
        th = TimeHorizon(num_periods=10, period_duration_h=1.0)
        dp = _flat_demand(100.0, 10)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = TimeWindowDecomposer(window_size=12, overlap=2)
        partitions = decomposer.partition(params)

        assert len(partitions) == 1
        assert partitions[0] is params

    def test_window_start_periods_correct(self) -> None:
        """Each window's start_period reflects its position in the horizon."""
        gens = [
            make_generator(id="g1", name="G1", capacity_mw=200.0,
                           fuel_type="coal", fuel_cost_per_mwh=30.0),
        ]
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(100.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = TimeWindowDecomposer(window_size=12, overlap=2)
        partitions = decomposer.partition(params)

        # step = 10
        assert partitions[0].time_horizon.start_period == 0
        assert partitions[1].time_horizon.start_period == 10
        assert partitions[2].time_horizon.start_period == 20

    def test_invalid_window_size_raises(self) -> None:
        """window_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            TimeWindowDecomposer(window_size=0, overlap=0)

    def test_invalid_overlap_raises(self) -> None:
        """overlap >= window_size raises ValueError."""
        with pytest.raises(ValueError, match="overlap.*must be < window_size"):
            TimeWindowDecomposer(window_size=6, overlap=6)

    def test_negative_overlap_raises(self) -> None:
        """Negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            TimeWindowDecomposer(window_size=6, overlap=-1)

    def test_zero_overlap_no_overlap_windows(self) -> None:
        """overlap=0 creates non-overlapping windows."""
        gens = [
            make_generator(id="g1", name="G1", capacity_mw=200.0,
                           fuel_type="coal", fuel_cost_per_mwh=30.0),
        ]
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(100.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = TimeWindowDecomposer(window_size=8, overlap=0)
        partitions = decomposer.partition(params)

        assert len(partitions) == 3  # 0-7, 8-15, 16-23
        assert partitions[0].time_horizon.num_periods == 8
        assert partitions[1].time_horizon.num_periods == 8
        assert partitions[2].time_horizon.num_periods == 8

    def test_missing_time_horizon_returns_empty(self) -> None:
        """Missing time_horizon returns empty partition list."""
        gens = [
            make_generator(id="g1", name="G1", capacity_mw=200.0,
                           fuel_type="coal", fuel_cost_per_mwh=30.0),
        ]
        dp = _flat_demand(100.0, 6)
        params = UCParameters(generators=gens, demand=dp)

        decomposer = TimeWindowDecomposer(window_size=4, overlap=1)
        partitions = decomposer.partition(params)

        assert partitions == []

    def test_missing_demand_returns_empty(self) -> None:
        """Missing demand returns empty partition list."""
        gens = [
            make_generator(id="g1", name="G1", capacity_mw=200.0,
                           fuel_type="coal", fuel_cost_per_mwh=30.0),
        ]
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        params = UCParameters(generators=gens, time_horizon=th)

        decomposer = TimeWindowDecomposer(window_size=4, overlap=1)
        partitions = decomposer.partition(params)

        assert partitions == []


# ======================================================================
# TestDecomposedSolve — feasibility of decomposed solves
# ======================================================================


class TestDecomposedSolveRegional:
    """Tests that regional decomposed solve produces a feasible schedule."""

    def test_regional_decomposed_solve_optimal(self) -> None:
        """Regional decomposed solve returns Optimal status."""
        gens = _make_regional_generators()
        th = TimeHorizon(num_periods=8, period_duration_h=1.0)
        # Low demand so each regional sub-problem is feasible
        dp = _flat_demand(100.0, 8)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        result = decomposer.solve_decomposed(params)

        assert result.status == "Optimal"
        assert result.total_cost > 0
        assert result.solve_time_s >= 0

    def test_regional_decomposed_has_all_generators(self) -> None:
        """Merged result contains schedules for all generators."""
        gens = _make_regional_generators()
        th = TimeHorizon(num_periods=8, period_duration_h=1.0)
        dp = _flat_demand(100.0, 8)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        result = decomposer.solve_decomposed(params)

        gen_ids_in_result = {s.generator_id for s in result.schedules}
        expected_ids = {g.id for g in gens}
        assert gen_ids_in_result == expected_ids, (
            f"Missing: {expected_ids - gen_ids_in_result}, "
            f"Extra: {gen_ids_in_result - expected_ids}"
        )

    def test_regional_decomposed_no_duplicates(self) -> None:
        """No generator appears twice in the merged result."""
        gens = _make_regional_generators()
        th = TimeHorizon(num_periods=8, period_duration_h=1.0)
        dp = _flat_demand(100.0, 8)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        result = decomposer.solve_decomposed(params)

        gen_ids = [s.generator_id for s in result.schedules]
        assert len(gen_ids) == len(set(gen_ids)), (
            f"Duplicate generator schedules found: {gen_ids}"
        )

    def test_regional_decomposed_schedule_lengths(self) -> None:
        """All schedules have correct length matching the time horizon."""
        gens = _make_regional_generators()
        num_periods = 8
        th = TimeHorizon(num_periods=num_periods, period_duration_h=1.0)
        dp = _flat_demand(100.0, num_periods)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        result = decomposer.solve_decomposed(params)

        for sched in result.schedules:
            assert len(sched.commitment) == num_periods, (
                f"Generator {sched.generator_id}: commitment length "
                f"{len(sched.commitment)} != {num_periods}"
            )
            assert len(sched.power_output_mw) == num_periods, (
                f"Generator {sched.generator_id}: power_output length "
                f"{len(sched.power_output_mw)} != {num_periods}"
            )


class TestDecomposedSolveFuelType:
    """Tests for fuel-type decomposed solve."""

    def test_fuel_type_decomposed_solve_optimal(self) -> None:
        """Fuel-type decomposed solve returns Optimal status."""
        gens = _make_fuel_type_generators()
        th = TimeHorizon(num_periods=8, period_duration_h=1.0)
        dp = _flat_demand(100.0, 8)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = FuelTypeDecomposer()
        result = decomposer.solve_decomposed(params)

        assert result.status == "Optimal"
        assert result.total_cost > 0

    def test_fuel_type_decomposed_all_generators_present(self) -> None:
        """Merged result contains schedules for every generator."""
        gens = _make_fuel_type_generators()
        th = TimeHorizon(num_periods=8, period_duration_h=1.0)
        dp = _flat_demand(100.0, 8)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = FuelTypeDecomposer()
        result = decomposer.solve_decomposed(params)

        gen_ids_in_result = {s.generator_id for s in result.schedules}
        expected_ids = {g.id for g in gens}
        assert gen_ids_in_result == expected_ids


class TestDecomposedSolveTimeWindow:
    """Tests for time-window decomposed solve."""

    def test_time_window_decomposed_solve_optimal(self) -> None:
        """Time-window decomposed solve returns Optimal status."""
        gens = [
            make_generator(
                id="g1",
                name="Base Gen",
                capacity_mw=200.0,
                fuel_type="coal",
                p_min_mw=50.0,
                fuel_cost_per_mwh=30.0,
                no_load_cost=50.0,
                startup_cost=1000.0,
            ),
        ]
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(100.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = TimeWindowDecomposer(window_size=12, overlap=2)
        result = decomposer.solve_decomposed(params)

        assert result.status == "Optimal"
        assert result.total_cost > 0

    def test_time_window_stitched_schedule_length(self) -> None:
        """Stitched schedule spans the full original time horizon."""
        gens = [
            make_generator(
                id="g1",
                name="Base Gen",
                capacity_mw=200.0,
                fuel_type="coal",
                p_min_mw=50.0,
                fuel_cost_per_mwh=30.0,
                no_load_cost=50.0,
                startup_cost=1000.0,
            ),
        ]
        num_periods = 24
        th = TimeHorizon(num_periods=num_periods, period_duration_h=1.0)
        dp = _flat_demand(100.0, num_periods)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = TimeWindowDecomposer(window_size=12, overlap=2)
        result = decomposer.solve_decomposed(params)

        assert len(result.schedules) == 1
        sched = result.schedules[0]
        assert len(sched.commitment) == num_periods, (
            f"Expected commitment length {num_periods}, "
            f"got {len(sched.commitment)}"
        )
        assert len(sched.power_output_mw) == num_periods, (
            f"Expected power_output length {num_periods}, "
            f"got {len(sched.power_output_mw)}"
        )

    def test_time_window_demand_met_at_each_period(self) -> None:
        """Stitched schedule meets demand at every period of the full horizon."""
        gens = [
            make_generator(
                id="g1",
                name="Base Gen",
                capacity_mw=300.0,
                fuel_type="coal",
                p_min_mw=50.0,
                fuel_cost_per_mwh=30.0,
                no_load_cost=50.0,
                startup_cost=1000.0,
            ),
        ]
        num_periods = 24
        demand_mw = 150.0
        th = TimeHorizon(num_periods=num_periods, period_duration_h=1.0)
        dp = _flat_demand(demand_mw, num_periods)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = TimeWindowDecomposer(window_size=12, overlap=2)
        result = decomposer.solve_decomposed(params)

        assert result.status == "Optimal"
        for t in range(num_periods):
            total_gen = sum(s.power_output_mw[t] for s in result.schedules)
            assert total_gen >= demand_mw - 1e-3, (
                f"Demand not met at t={t}: "
                f"generation={total_gen:.4f} < demand={demand_mw}"
            )

    def test_time_window_boundary_consistency(self) -> None:
        """At window boundaries, stitched commitment values are valid (0 or 1)."""
        gens = [
            make_generator(
                id="g1",
                name="Base Gen",
                capacity_mw=200.0,
                fuel_type="coal",
                p_min_mw=50.0,
                fuel_cost_per_mwh=30.0,
                no_load_cost=50.0,
                startup_cost=1000.0,
            ),
        ]
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(100.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = TimeWindowDecomposer(window_size=12, overlap=2)
        result = decomposer.solve_decomposed(params)

        sched = result.schedules[0]
        # Check boundary periods (step=10, so boundaries at t=9/10, t=19/20)
        for t in range(len(sched.commitment)):
            assert sched.commitment[t] in (0, 1), (
                f"Invalid commitment at t={t}: {sched.commitment[t]}"
            )
            assert sched.power_output_mw[t] >= -1e-6, (
                f"Negative power at t={t}: {sched.power_output_mw[t]}"
            )


# ======================================================================
# TestEmptyPartitionHandling
# ======================================================================


class TestEmptyPartitionHandling:
    """Tests for graceful handling of empty partitions."""

    def test_empty_generators_regional_not_solved(self) -> None:
        """Regional decomposer with no generators returns Not Solved."""
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(100.0, 6)
        params = UCParameters(generators=[], demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        result = decomposer.solve_decomposed(params)

        assert result.status == "Not Solved"
        assert len(result.warnings) > 0

    def test_empty_generators_fuel_type_not_solved(self) -> None:
        """Fuel-type decomposer with no generators returns Not Solved."""
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(100.0, 6)
        params = UCParameters(generators=[], demand=dp, time_horizon=th)

        decomposer = FuelTypeDecomposer()
        result = decomposer.solve_decomposed(params)

        assert result.status == "Not Solved"
        assert len(result.warnings) > 0

    def test_time_window_missing_horizon_not_solved(self) -> None:
        """TimeWindowDecomposer with no time_horizon returns Not Solved."""
        gens = [
            make_generator(id="g1", name="G1", capacity_mw=200.0,
                           fuel_type="coal", fuel_cost_per_mwh=30.0),
        ]
        dp = _flat_demand(100.0, 6)
        params = UCParameters(generators=gens, demand=dp)

        decomposer = TimeWindowDecomposer(window_size=4, overlap=1)
        result = decomposer.solve_decomposed(params)

        assert result.status == "Not Solved"


# ======================================================================
# TestResultMerging
# ======================================================================


class TestResultMerging:
    """Tests that result merging produces correct totals."""

    def test_merged_total_cost_is_sum_of_partition_costs(self) -> None:
        """Merged total_cost equals the sum of individual generator costs."""
        gens = _make_regional_generators()
        th = TimeHorizon(num_periods=8, period_duration_h=1.0)
        dp = _flat_demand(100.0, 8)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        result = decomposer.solve_decomposed(params)

        assert result.status == "Optimal"
        gen_cost_sum = sum(s.total_cost for s in result.schedules)
        assert abs(result.total_cost - gen_cost_sum) < 1.0, (
            f"total_cost={result.total_cost:.2f} != "
            f"sum(gen_costs)={gen_cost_sum:.2f}"
        )

    def test_merged_warnings_aggregated(self) -> None:
        """Warnings from all sub-problems are aggregated in merged result."""
        decomposer = RegionalDecomposer()

        # Create results with warnings
        r1 = UCResult(
            status="Optimal",
            total_cost=100.0,
            warnings=["Warning from region A"],
        )
        r2 = UCResult(
            status="Optimal",
            total_cost=200.0,
            warnings=["Warning from region B"],
        )

        merged = decomposer.merge_results([r1, r2])

        assert len(merged.warnings) == 2
        assert "Warning from region A" in merged.warnings
        assert "Warning from region B" in merged.warnings

    def test_merged_status_optimal_when_all_optimal(self) -> None:
        """Status is Optimal when all sub-problems are Optimal."""
        decomposer = RegionalDecomposer()

        r1 = UCResult(status="Optimal", total_cost=100.0)
        r2 = UCResult(status="Optimal", total_cost=200.0)

        merged = decomposer.merge_results([r1, r2])
        assert merged.status == "Optimal"

    def test_merged_status_infeasible_if_any_infeasible(self) -> None:
        """Status is Infeasible if any sub-problem is Infeasible."""
        decomposer = RegionalDecomposer()

        r1 = UCResult(status="Optimal", total_cost=100.0)
        r2 = UCResult(status="Infeasible", total_cost=0.0)

        merged = decomposer.merge_results([r1, r2])
        assert merged.status == "Infeasible"

    def test_merged_gap_is_worst_case(self) -> None:
        """Merged gap is the maximum gap across sub-problems."""
        decomposer = RegionalDecomposer()

        r1 = UCResult(status="Optimal", total_cost=100.0, gap=0.001)
        r2 = UCResult(status="Optimal", total_cost=200.0, gap=0.05)

        merged = decomposer.merge_results([r1, r2])
        assert merged.gap == 0.05

    def test_merge_empty_results_not_solved(self) -> None:
        """Merging empty results list returns Not Solved."""
        decomposer = RegionalDecomposer()

        merged = decomposer.merge_results([])
        assert merged.status == "Not Solved"
        assert len(merged.warnings) > 0

    def test_time_window_merged_cost_consistent(self) -> None:
        """Time-window decomposed result cost is recomputed from stitched schedule."""
        gens = [
            make_generator(
                id="g1",
                name="Base Gen",
                capacity_mw=200.0,
                fuel_type="coal",
                p_min_mw=50.0,
                fuel_cost_per_mwh=30.0,
                no_load_cost=50.0,
                labor_cost_per_h=10.0,
                startup_cost=1000.0,
            ),
        ]
        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(100.0, 24)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = TimeWindowDecomposer(window_size=12, overlap=2)
        result = decomposer.solve_decomposed(params)

        assert result.status == "Optimal"

        # Verify cost consistency: total_cost == sum of generator schedule costs
        gen_cost_sum = sum(s.total_cost for s in result.schedules)
        assert abs(result.total_cost - gen_cost_sum) < 1.0

        # Verify individual cost components are reasonable
        sched = result.schedules[0]
        assert sched.fuel_cost >= 0
        assert sched.no_load_cost >= 0
        assert sched.startup_cost >= 0


# ======================================================================
# TestFactoryFunction
# ======================================================================


class TestCreateDecomposer:
    """Tests for the create_decomposer factory function."""

    def test_create_regional(self) -> None:
        """create_decomposer('regional') returns RegionalDecomposer."""
        d = create_decomposer("regional")
        assert isinstance(d, RegionalDecomposer)

    def test_create_fuel_type(self) -> None:
        """create_decomposer('fuel_type') returns FuelTypeDecomposer."""
        d = create_decomposer("fuel_type")
        assert isinstance(d, FuelTypeDecomposer)

    def test_create_time_window(self) -> None:
        """create_decomposer('time_window') returns TimeWindowDecomposer."""
        d = create_decomposer("time_window")
        assert isinstance(d, TimeWindowDecomposer)

    def test_create_time_window_with_kwargs(self) -> None:
        """create_decomposer passes kwargs to TimeWindowDecomposer."""
        d = create_decomposer("time_window", window_size=8, overlap=3)
        assert isinstance(d, TimeWindowDecomposer)
        assert d.window_size == 8
        assert d.overlap == 3

    def test_create_unknown_strategy_raises(self) -> None:
        """Unknown strategy name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown decomposition strategy"):
            create_decomposer("unknown_strategy")

    def test_create_case_insensitive(self) -> None:
        """Strategy names are case-insensitive."""
        d = create_decomposer("Regional")
        assert isinstance(d, RegionalDecomposer)

        d2 = create_decomposer("FUEL_TYPE")
        assert isinstance(d2, FuelTypeDecomposer)


# ======================================================================
# TestConsistencyAndIntegrity
# ======================================================================


class TestConsistencyAndIntegrity:
    """Integration-level tests verifying decomposition correctness."""

    def test_regional_vs_direct_solve_all_generators_present(self) -> None:
        """Regional decomposed result covers same generators as direct solve."""
        gens = _make_regional_generators()
        th = TimeHorizon(num_periods=8, period_duration_h=1.0)
        dp = _flat_demand(100.0, 8)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        # Direct solve
        direct_result = solve_uc(params)

        # Decomposed solve
        decomposer = RegionalDecomposer()
        decomposed_result = decomposer.solve_decomposed(params)

        direct_gen_ids = {s.generator_id for s in direct_result.schedules}
        decomposed_gen_ids = {s.generator_id for s in decomposed_result.schedules}

        assert direct_gen_ids == decomposed_gen_ids

    def test_time_window_with_multiple_generators(self) -> None:
        """Time-window decomposition handles multiple generators correctly."""
        gens = [
            make_generator(
                id="g1",
                name="Base Coal",
                capacity_mw=200.0,
                fuel_type="coal",
                p_min_mw=50.0,
                fuel_cost_per_mwh=30.0,
                no_load_cost=100.0,
                startup_cost=5000.0,
            ),
            make_generator(
                id="g2",
                name="Peaker LNG",
                capacity_mw=100.0,
                fuel_type="lng",
                p_min_mw=20.0,
                fuel_cost_per_mwh=50.0,
                no_load_cost=50.0,
                startup_cost=2000.0,
            ),
        ]
        th = TimeHorizon(num_periods=20, period_duration_h=1.0)
        dp = _flat_demand(150.0, 20)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = TimeWindowDecomposer(window_size=10, overlap=2)
        result = decomposer.solve_decomposed(params)

        assert result.status == "Optimal"

        # Both generators should have schedules
        gen_ids = {s.generator_id for s in result.schedules}
        assert gen_ids == {"g1", "g2"}

        # Schedules should span the full horizon
        for sched in result.schedules:
            assert len(sched.commitment) == 20
            assert len(sched.power_output_mw) == 20

    def test_three_region_decomposition(self) -> None:
        """Decomposition works with 3 regions."""
        gens = [
            make_generator(
                id="g_r1",
                name="Region 1 Gen",
                capacity_mw=200.0,
                fuel_type="coal",
                region="tohoku",
                p_min_mw=50.0,
                fuel_cost_per_mwh=30.0,
                no_load_cost=50.0,
            ),
            make_generator(
                id="g_r2",
                name="Region 2 Gen",
                capacity_mw=150.0,
                fuel_type="lng",
                region="kanto",
                p_min_mw=30.0,
                fuel_cost_per_mwh=50.0,
                no_load_cost=30.0,
            ),
            make_generator(
                id="g_r3",
                name="Region 3 Gen",
                capacity_mw=100.0,
                fuel_type="oil",
                region="chubu",
                p_min_mw=10.0,
                fuel_cost_per_mwh=80.0,
                no_load_cost=20.0,
            ),
        ]
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(100.0, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)
        assert len(partitions) == 3

        result = decomposer.solve_decomposed(params)
        assert result.status == "Optimal"
        assert {s.generator_id for s in result.schedules} == {
            "g_r1", "g_r2", "g_r3"
        }

    def test_unassigned_region_generators_grouped_together(self) -> None:
        """Generators without a region are grouped under '_unassigned'."""
        gens = [
            make_generator(
                id="g_named",
                name="Named Region",
                capacity_mw=200.0,
                fuel_type="coal",
                region="shikoku",
                p_min_mw=50.0,
                fuel_cost_per_mwh=30.0,
                no_load_cost=50.0,
            ),
            make_generator(
                id="g_no_region",
                name="No Region",
                capacity_mw=100.0,
                fuel_type="lng",
                region="",
                p_min_mw=20.0,
                fuel_cost_per_mwh=50.0,
                no_load_cost=30.0,
            ),
        ]
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(100.0, 6)
        params = UCParameters(generators=gens, demand=dp, time_horizon=th)

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)

        assert len(partitions) == 2
        all_ids = _collect_gen_ids(partitions)
        assert all_ids == {"g_named", "g_no_region"}
