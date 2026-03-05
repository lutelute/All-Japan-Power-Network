"""Tests for interconnection data models, loader, and constraint builders.

Tests cover:
- Interconnection dataclass validation (empty id, negative capacity, same from/to region)
- InterconnectionLoader loads all 9 records from YAML
- InterconnectionLoader raises FileNotFoundError for missing file
- InterconnectionFlow dataclass construction
- Interconnection and InterconnectionFlow exports from src.uc
- Transmission capacity constraint builder (flow bounded by capacity)
- Nodal balance constraint builder (per-region generation + net flow >= demand)
- Flow sign convention (positive = from_region -> to_region)
- Two-region integration test via solve_uc with interconnection flow verification
- Regression test: solve_uc without interconnections produces unchanged behavior
- Decomposition fallback: RegionalDecomposer returns single partition with interconnections
- Decomposition unchanged: RegionalDecomposer decomposes normally without interconnections
- Config toggle: interconnection.enabled=false prevents loading/use
- Decomposed solve with interconnections falls back to national MILP
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pulp
import pytest
import yaml

from src.model.generator import Generator
from src.uc.constraints import (
    add_nodal_balance_constraints,
    add_transmission_capacity_constraints,
)
from src.uc.decomposition import RegionalDecomposer
from src.uc.interconnection_loader import InterconnectionLoader
from src.uc.models import (
    DemandProfile,
    Interconnection,
    InterconnectionFlow,
    TimeHorizon,
    UCParameters,
)
from src.uc.solver import solve_uc
from tests.conftest import make_generator, make_interconnection


# ======================================================================
# TestInterconnectionDataclass — validation
# ======================================================================


class TestInterconnectionDataclass:
    """Tests that the Interconnection dataclass validates inputs correctly."""

    def test_valid_interconnection(self) -> None:
        """A valid Interconnection instance is created without errors."""
        ic = make_interconnection()
        assert ic.id == "ic_test_001"
        assert ic.name_en == "Test Interconnection"
        assert ic.from_region == "tokyo"
        assert ic.to_region == "chubu"
        assert ic.capacity_mw == 1000.0
        assert ic.type == "AC"

    def test_empty_id_raises_value_error(self) -> None:
        """Empty id raises ValueError."""
        with pytest.raises(ValueError, match="id must not be empty"):
            make_interconnection(id="")

    def test_negative_capacity_raises_value_error(self) -> None:
        """Negative capacity_mw raises ValueError."""
        with pytest.raises(ValueError, match="capacity_mw must be positive"):
            make_interconnection(capacity_mw=-100.0)

    def test_zero_capacity_raises_value_error(self) -> None:
        """Zero capacity_mw raises ValueError."""
        with pytest.raises(ValueError, match="capacity_mw must be positive"):
            make_interconnection(capacity_mw=0.0)

    def test_same_from_to_region_raises_value_error(self) -> None:
        """Same from_region and to_region raises ValueError."""
        with pytest.raises(ValueError, match="from_region and to_region must differ"):
            make_interconnection(from_region="tokyo", to_region="tokyo")

    def test_custom_type_hvdc(self) -> None:
        """HVDC type is accepted."""
        ic = make_interconnection(type="HVDC")
        assert ic.type == "HVDC"

    def test_custom_type_fc(self) -> None:
        """FC (frequency converter) type is accepted."""
        ic = make_interconnection(type="FC")
        assert ic.type == "FC"

    def test_default_type_is_ac(self) -> None:
        """Default type is AC when not specified."""
        ic = Interconnection(
            id="ic_test",
            name_en="Test",
            from_region="hokkaido",
            to_region="tohoku",
            capacity_mw=500.0,
        )
        assert ic.type == "AC"


# ======================================================================
# TestInterconnectionFlow — dataclass construction
# ======================================================================


class TestInterconnectionFlow:
    """Tests for the InterconnectionFlow dataclass."""

    def test_default_construction(self) -> None:
        """InterconnectionFlow defaults to empty id and empty flow list."""
        flow = InterconnectionFlow()
        assert flow.interconnection_id == ""
        assert flow.flow_mw == []

    def test_construction_with_values(self) -> None:
        """InterconnectionFlow stores provided values correctly."""
        flow = InterconnectionFlow(
            interconnection_id="ic_001",
            flow_mw=[100.0, 200.0, -50.0],
        )
        assert flow.interconnection_id == "ic_001"
        assert flow.flow_mw == [100.0, 200.0, -50.0]
        assert len(flow.flow_mw) == 3


# ======================================================================
# TestInterconnectionLoader — YAML loading
# ======================================================================


class TestInterconnectionLoader:
    """Tests for the InterconnectionLoader class."""

    def test_load_all_9_interconnections(self, project_root: Path) -> None:
        """Loader reads all 9 interconnection records from YAML."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        assert len(interconnections) == 9

    def test_loaded_records_are_interconnection_instances(
        self, project_root: Path
    ) -> None:
        """All loaded records are Interconnection dataclass instances."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        for ic in interconnections:
            assert isinstance(ic, Interconnection)

    def test_first_record_is_hokkaido_honshu_hvdc(
        self, project_root: Path
    ) -> None:
        """First interconnection is the Hokkaido-Honshu HVDC link."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        ic = interconnections[0]
        assert ic.id == "ic_001"
        assert ic.name_en == "Hokkaido-Honshu HVDC Link"
        assert ic.from_region == "hokkaido"
        assert ic.to_region == "tohoku"
        assert ic.capacity_mw == 900.0
        assert ic.type == "HVDC"

    def test_tokyo_chubu_fc_interconnection(self, project_root: Path) -> None:
        """Tokyo-Chubu FC interconnection has correct capacity and type."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        # ic_003 is Tokyo-Chubu FC
        ic_003 = [ic for ic in interconnections if ic.id == "ic_003"][0]
        assert ic_003.from_region == "tokyo"
        assert ic_003.to_region == "chubu"
        assert ic_003.capacity_mw == 2100.0
        assert ic_003.type == "FC"

    def test_all_ids_unique(self, project_root: Path) -> None:
        """All interconnection IDs are unique."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        ids = [ic.id for ic in interconnections]
        assert len(ids) == len(set(ids))

    def test_all_capacities_positive(self, project_root: Path) -> None:
        """All interconnection capacities are positive."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        for ic in interconnections:
            assert ic.capacity_mw > 0, f"{ic.id} has non-positive capacity"

    def test_all_from_to_regions_differ(self, project_root: Path) -> None:
        """No interconnection has the same from_region and to_region."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        for ic in interconnections:
            assert ic.from_region != ic.to_region, (
                f"{ic.id} has same from/to region: {ic.from_region}"
            )

    def test_missing_file_raises_file_not_found_error(self) -> None:
        """Loader raises FileNotFoundError for a non-existent YAML path."""
        loader = InterconnectionLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/interconnections.yaml")


# ======================================================================
# TestTransmissionCapacityConstraints — constraint builder
# ======================================================================


class TestTransmissionCapacityConstraints:
    """Tests for add_transmission_capacity_constraints() builder."""

    def test_constraint_count_1ic_3timesteps(self) -> None:
        """1 interconnection x 3 timesteps produces 6 constraints (2 per timestep)."""
        model = pulp.LpProblem("test_tx_cap", pulp.LpMinimize)
        ic = make_interconnection(id="ic_ab", capacity_mw=500.0)
        timesteps = [0, 1, 2]

        # Create flow variables (continuous, unbounded)
        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for t in timesteps:
            f[(ic.id, t)] = pulp.LpVariable(f"f_{ic.id}_{t}", cat="Continuous")

        count = add_transmission_capacity_constraints(
            model, f, [ic], timesteps
        )

        assert count == 6  # 1 IC x 3 timesteps x 2 (ub + lb)

    def test_constraint_count_2ic_4timesteps(self) -> None:
        """2 interconnections x 4 timesteps produces 16 constraints."""
        model = pulp.LpProblem("test_tx_cap_2ic", pulp.LpMinimize)
        ic1 = make_interconnection(id="ic_01", from_region="a", to_region="b", capacity_mw=300.0)
        ic2 = make_interconnection(id="ic_02", from_region="b", to_region="c", capacity_mw=200.0)
        timesteps = [0, 1, 2, 3]

        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for ic in [ic1, ic2]:
            for t in timesteps:
                f[(ic.id, t)] = pulp.LpVariable(f"f_{ic.id}_{t}", cat="Continuous")

        count = add_transmission_capacity_constraints(
            model, f, [ic1, ic2], timesteps
        )

        assert count == 16  # 2 IC x 4 timesteps x 2

    def test_constraint_names_follow_convention(self) -> None:
        """Constraint names follow tx_cap_ub_{ic_id}_t{t} / tx_cap_lb_{ic_id}_t{t}."""
        model = pulp.LpProblem("test_names", pulp.LpMinimize)
        ic = make_interconnection(id="ic_test", capacity_mw=100.0)
        timesteps = [0, 1]

        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for t in timesteps:
            f[(ic.id, t)] = pulp.LpVariable(f"f_{ic.id}_{t}", cat="Continuous")

        add_transmission_capacity_constraints(model, f, [ic], timesteps)

        constraint_names = set(model.constraints.keys())
        assert "tx_cap_ub_ic_test_t0" in constraint_names
        assert "tx_cap_lb_ic_test_t0" in constraint_names
        assert "tx_cap_ub_ic_test_t1" in constraint_names
        assert "tx_cap_lb_ic_test_t1" in constraint_names

    def test_flow_within_upper_bound(self) -> None:
        """Maximising flow is limited by capacity upper bound."""
        model = pulp.LpProblem("test_ub", pulp.LpMaximize)
        ic = make_interconnection(id="ic_ub", capacity_mw=500.0)
        timesteps = [0]

        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        f[(ic.id, 0)] = pulp.LpVariable("f_ic_ub_0", cat="Continuous")

        # Maximise f => should hit upper bound at 500
        model += f[(ic.id, 0)]
        add_transmission_capacity_constraints(model, f, [ic], timesteps)

        model.solve(pulp.PULP_CBC_CMD(msg=0))
        assert pulp.LpStatus[model.status] == "Optimal"
        assert abs(pulp.value(f[(ic.id, 0)]) - 500.0) < 1e-3

    def test_flow_within_lower_bound(self) -> None:
        """Minimising flow is limited by capacity lower bound (negative)."""
        model = pulp.LpProblem("test_lb", pulp.LpMinimize)
        ic = make_interconnection(id="ic_lb", capacity_mw=500.0)
        timesteps = [0]

        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        f[(ic.id, 0)] = pulp.LpVariable("f_ic_lb_0", cat="Continuous")

        # Minimise f => should hit lower bound at -500
        model += f[(ic.id, 0)]
        add_transmission_capacity_constraints(model, f, [ic], timesteps)

        model.solve(pulp.PULP_CBC_CMD(msg=0))
        assert pulp.LpStatus[model.status] == "Optimal"
        assert abs(pulp.value(f[(ic.id, 0)]) - (-500.0)) < 1e-3

    def test_solve_multiple_timesteps_flow_bounded(self) -> None:
        """Flow variables at all timesteps are bounded by capacity after solve."""
        model = pulp.LpProblem("test_multi_t", pulp.LpMaximize)
        ic = make_interconnection(id="ic_multi", capacity_mw=300.0)
        timesteps = [0, 1, 2]

        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for t in timesteps:
            f[(ic.id, t)] = pulp.LpVariable(f"f_ic_multi_{t}", cat="Continuous")

        # Maximise sum of flows
        model += pulp.lpSum(f[(ic.id, t)] for t in timesteps)
        add_transmission_capacity_constraints(model, f, [ic], timesteps)

        model.solve(pulp.PULP_CBC_CMD(msg=0))
        assert pulp.LpStatus[model.status] == "Optimal"

        for t in timesteps:
            flow_val = pulp.value(f[(ic.id, t)])
            assert flow_val <= 300.0 + 1e-3, (
                f"Flow at t={t} exceeds upper bound: {flow_val}"
            )
            assert flow_val >= -300.0 - 1e-3, (
                f"Flow at t={t} below lower bound: {flow_val}"
            )


# ======================================================================
# TestNodalBalanceConstraints — constraint builder
# ======================================================================


class TestNodalBalanceConstraints:
    """Tests for add_nodal_balance_constraints() builder."""

    def test_constraint_count_2regions_3timesteps(self) -> None:
        """2 regions x 3 timesteps produces 6 nodal balance constraints."""
        model = pulp.LpProblem("test_nodal", pulp.LpMinimize)

        # 2 generators in different regions
        g_a = make_generator(id="g_a", name="Gen A", region="region_a", capacity_mw=200.0)
        g_b = make_generator(id="g_b", name="Gen B", region="region_b", capacity_mw=200.0)
        generators = [g_a, g_b]

        # 1 interconnection: region_a -> region_b
        ic = make_interconnection(
            id="ic_ab", from_region="region_a", to_region="region_b", capacity_mw=100.0,
        )

        timesteps = [0, 1, 2]

        # Create power and flow variables
        p: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for g in generators:
            for t in timesteps:
                p[(g.id, t)] = pulp.LpVariable(
                    f"p_{g.id}_{t}", lowBound=0, upBound=g.capacity_mw,
                )

        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for t in timesteps:
            f[(ic.id, t)] = pulp.LpVariable(f"f_{ic.id}_{t}", cat="Continuous")

        regional_demand = {
            "region_a": [100.0, 100.0, 100.0],
            "region_b": [100.0, 100.0, 100.0],
        }

        count = add_nodal_balance_constraints(
            model, p, f, generators, [ic], timesteps, regional_demand,
        )

        assert count == 6  # 2 regions x 3 timesteps

    def test_constraint_names_follow_convention(self) -> None:
        """Constraint names follow nodal_bal_{region}_t{t} convention."""
        model = pulp.LpProblem("test_names", pulp.LpMinimize)

        g_a = make_generator(id="g_a", name="Gen A", region="tokyo", capacity_mw=200.0)
        g_b = make_generator(id="g_b", name="Gen B", region="chubu", capacity_mw=200.0)

        ic = make_interconnection(
            id="ic_tc", from_region="tokyo", to_region="chubu", capacity_mw=100.0,
        )
        timesteps = [0, 1]

        p: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for g in [g_a, g_b]:
            for t in timesteps:
                p[(g.id, t)] = pulp.LpVariable(
                    f"p_{g.id}_{t}", lowBound=0, upBound=g.capacity_mw,
                )

        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for t in timesteps:
            f[(ic.id, t)] = pulp.LpVariable(f"f_{ic.id}_{t}", cat="Continuous")

        regional_demand = {
            "tokyo": [100.0, 100.0],
            "chubu": [100.0, 100.0],
        }

        add_nodal_balance_constraints(
            model, p, f, [g_a, g_b], [ic], timesteps, regional_demand,
        )

        constraint_names = set(model.constraints.keys())
        assert "nodal_bal_tokyo_t0" in constraint_names
        assert "nodal_bal_tokyo_t1" in constraint_names
        assert "nodal_bal_chubu_t0" in constraint_names
        assert "nodal_bal_chubu_t1" in constraint_names

    def test_solve_regions_meet_demand(self) -> None:
        """Both regions meet demand after solve with interconnection."""
        model = pulp.LpProblem("test_demand_met", pulp.LpMinimize)

        # Region A: 300 MW generator, demand 100 MW (surplus)
        # Region B: 100 MW generator, demand 150 MW (deficit of 50 MW)
        g_a = make_generator(
            id="g_a", name="Gen A", region="region_a", capacity_mw=300.0,
            fuel_cost_per_mwh=30.0,
        )
        g_b = make_generator(
            id="g_b", name="Gen B", region="region_b", capacity_mw=100.0,
            fuel_cost_per_mwh=50.0,
        )
        generators = [g_a, g_b]

        # Interconnection allows 200 MW from A -> B
        ic = make_interconnection(
            id="ic_ab", from_region="region_a", to_region="region_b", capacity_mw=200.0,
        )

        timesteps = [0]

        p: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for g in generators:
            for t in timesteps:
                p[(g.id, t)] = pulp.LpVariable(
                    f"p_{g.id}_{t}", lowBound=0, upBound=g.capacity_mw,
                )

        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for t in timesteps:
            f[(ic.id, t)] = pulp.LpVariable(f"f_{ic.id}_{t}", cat="Continuous")

        regional_demand = {
            "region_a": [100.0],
            "region_b": [150.0],
        }

        # Objective: minimise generation cost
        model += pulp.lpSum(
            g.fuel_cost_per_mwh * p[(g.id, t)] for g in generators for t in timesteps
        )

        add_nodal_balance_constraints(
            model, p, f, generators, [ic], timesteps, regional_demand,
        )
        add_transmission_capacity_constraints(model, f, [ic], timesteps)

        model.solve(pulp.PULP_CBC_CMD(msg=0))
        assert pulp.LpStatus[model.status] == "Optimal"

        # Verify region A demand met: gen_a - export >= 100
        gen_a_val = pulp.value(p[("g_a", 0)])
        flow_val = pulp.value(f[("ic_ab", 0)])
        assert gen_a_val - flow_val >= 100.0 - 1e-3, (
            f"Region A balance violated: gen={gen_a_val}, flow_out={flow_val}"
        )

        # Verify region B demand met: gen_b + import >= 150
        gen_b_val = pulp.value(p[("g_b", 0)])
        assert gen_b_val + flow_val >= 150.0 - 1e-3, (
            f"Region B balance violated: gen={gen_b_val}, flow_in={flow_val}"
        )

    def test_infeasible_when_capacity_insufficient(self) -> None:
        """Model is infeasible when interconnection capacity is too small."""
        model = pulp.LpProblem("test_infeasible", pulp.LpMinimize)

        # Region A: 300 MW gen, demand 50 MW
        # Region B: 50 MW gen, demand 200 MW (deficit 150 MW, IC cap only 100)
        g_a = make_generator(
            id="g_a", name="Gen A", region="region_a", capacity_mw=300.0,
        )
        g_b = make_generator(
            id="g_b", name="Gen B", region="region_b", capacity_mw=50.0,
        )
        generators = [g_a, g_b]

        ic = make_interconnection(
            id="ic_ab", from_region="region_a", to_region="region_b", capacity_mw=100.0,
        )

        timesteps = [0]

        p: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for g in generators:
            for t in timesteps:
                p[(g.id, t)] = pulp.LpVariable(
                    f"p_{g.id}_{t}", lowBound=0, upBound=g.capacity_mw,
                )

        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for t in timesteps:
            f[(ic.id, t)] = pulp.LpVariable(f"f_{ic.id}_{t}", cat="Continuous")

        regional_demand = {
            "region_a": [50.0],
            "region_b": [200.0],  # needs 150 from IC but cap is 100
        }

        model += 0  # dummy objective
        add_nodal_balance_constraints(
            model, p, f, generators, [ic], timesteps, regional_demand,
        )
        add_transmission_capacity_constraints(model, f, [ic], timesteps)

        model.solve(pulp.PULP_CBC_CMD(msg=0))
        assert pulp.LpStatus[model.status] == "Infeasible"

    def test_multiple_generators_per_region(self) -> None:
        """Nodal balance correctly sums multiple generators in one region."""
        model = pulp.LpProblem("test_multi_gen", pulp.LpMinimize)

        # Region A: 2 generators (100 MW each), demand 180 MW
        g_a1 = make_generator(id="g_a1", name="Gen A1", region="region_a", capacity_mw=100.0)
        g_a2 = make_generator(id="g_a2", name="Gen A2", region="region_a", capacity_mw=100.0)
        # Region B: 1 generator (200 MW), demand 50 MW
        g_b = make_generator(id="g_b", name="Gen B", region="region_b", capacity_mw=200.0)
        generators = [g_a1, g_a2, g_b]

        ic = make_interconnection(
            id="ic_ab", from_region="region_a", to_region="region_b", capacity_mw=50.0,
        )

        timesteps = [0]

        p: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for g in generators:
            for t in timesteps:
                p[(g.id, t)] = pulp.LpVariable(
                    f"p_{g.id}_{t}", lowBound=0, upBound=g.capacity_mw,
                )

        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for t in timesteps:
            f[(ic.id, t)] = pulp.LpVariable(f"f_{ic.id}_{t}", cat="Continuous")

        regional_demand = {
            "region_a": [180.0],
            "region_b": [50.0],
        }

        model += 0  # dummy objective
        add_nodal_balance_constraints(
            model, p, f, generators, [ic], timesteps, regional_demand,
        )
        add_transmission_capacity_constraints(model, f, [ic], timesteps)

        model.solve(pulp.PULP_CBC_CMD(msg=0))
        assert pulp.LpStatus[model.status] == "Optimal"

        # Verify region A: g_a1 + g_a2 - export >= 180
        gen_a_total = pulp.value(p[("g_a1", 0)]) + pulp.value(p[("g_a2", 0)])
        flow_val = pulp.value(f[("ic_ab", 0)])
        assert gen_a_total - flow_val >= 180.0 - 1e-3


# ======================================================================
# TestFlowSignConvention — positive = from_region -> to_region
# ======================================================================


class TestFlowSignConvention:
    """Tests that positive flow represents power from from_region to to_region."""

    def test_flow_sign_positive_from_surplus_to_deficit(self) -> None:
        """Positive flow goes from from_region (surplus) to to_region (deficit)."""
        model = pulp.LpProblem("test_sign", pulp.LpMinimize)

        # Region A (from_region): 300 MW gen, demand 100 MW (surplus)
        # Region B (to_region): 50 MW gen, demand 150 MW (deficit)
        g_a = make_generator(
            id="g_a", name="Gen A", region="region_a", capacity_mw=300.0,
            fuel_cost_per_mwh=10.0,
        )
        g_b = make_generator(
            id="g_b", name="Gen B", region="region_b", capacity_mw=50.0,
            fuel_cost_per_mwh=10.0,
        )
        generators = [g_a, g_b]

        # IC from region_a -> region_b
        ic = make_interconnection(
            id="ic_ab", from_region="region_a", to_region="region_b", capacity_mw=200.0,
        )

        timesteps = [0]

        p: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for g in generators:
            for t in timesteps:
                p[(g.id, t)] = pulp.LpVariable(
                    f"p_{g.id}_{t}", lowBound=0, upBound=g.capacity_mw,
                )

        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for t in timesteps:
            f[(ic.id, t)] = pulp.LpVariable(f"f_{ic.id}_{t}", cat="Continuous")

        regional_demand = {
            "region_a": [100.0],
            "region_b": [150.0],
        }

        # Minimise total generation cost
        model += pulp.lpSum(
            g.fuel_cost_per_mwh * p[(g.id, t)] for g in generators for t in timesteps
        )

        add_nodal_balance_constraints(
            model, p, f, generators, [ic], timesteps, regional_demand,
        )
        add_transmission_capacity_constraints(model, f, [ic], timesteps)

        model.solve(pulp.PULP_CBC_CMD(msg=0))
        assert pulp.LpStatus[model.status] == "Optimal"

        # Flow should be positive: power goes from region_a to region_b
        flow_val = pulp.value(f[("ic_ab", 0)])
        assert flow_val > 0, (
            f"Expected positive flow from region_a to region_b, got {flow_val}"
        )

    def test_flow_sign_negative_when_reverse_direction(self) -> None:
        """Negative flow means power goes from to_region to from_region (reverse)."""
        model = pulp.LpProblem("test_neg_sign", pulp.LpMinimize)

        # Region A (from_region): 50 MW gen, demand 150 MW (deficit)
        # Region B (to_region): 300 MW gen, demand 100 MW (surplus)
        g_a = make_generator(
            id="g_a", name="Gen A", region="region_a", capacity_mw=50.0,
            fuel_cost_per_mwh=10.0,
        )
        g_b = make_generator(
            id="g_b", name="Gen B", region="region_b", capacity_mw=300.0,
            fuel_cost_per_mwh=10.0,
        )
        generators = [g_a, g_b]

        # IC still defined as from region_a -> region_b
        ic = make_interconnection(
            id="ic_ab", from_region="region_a", to_region="region_b", capacity_mw=200.0,
        )

        timesteps = [0]

        p: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for g in generators:
            for t in timesteps:
                p[(g.id, t)] = pulp.LpVariable(
                    f"p_{g.id}_{t}", lowBound=0, upBound=g.capacity_mw,
                )

        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for t in timesteps:
            f[(ic.id, t)] = pulp.LpVariable(f"f_{ic.id}_{t}", cat="Continuous")

        regional_demand = {
            "region_a": [150.0],
            "region_b": [100.0],
        }

        # Minimise total generation cost
        model += pulp.lpSum(
            g.fuel_cost_per_mwh * p[(g.id, t)] for g in generators for t in timesteps
        )

        add_nodal_balance_constraints(
            model, p, f, generators, [ic], timesteps, regional_demand,
        )
        add_transmission_capacity_constraints(model, f, [ic], timesteps)

        model.solve(pulp.PULP_CBC_CMD(msg=0))
        assert pulp.LpStatus[model.status] == "Optimal"

        # Flow should be negative: power goes from region_b to region_a (reverse)
        flow_val = pulp.value(f[("ic_ab", 0)])
        assert flow_val < 0, (
            f"Expected negative flow (reverse direction), got {flow_val}"
        )

    def test_flow_sign_magnitude_matches_deficit(self) -> None:
        """Flow magnitude matches the deficit that needs to be transferred."""
        model = pulp.LpProblem("test_magnitude", pulp.LpMinimize)

        # Region A: 300 MW gen, demand 100 MW
        # Region B: 0 MW gen capacity (no gen), demand 80 MW
        # Flow from A to B should be exactly 80 MW
        g_a = make_generator(
            id="g_a", name="Gen A", region="region_a", capacity_mw=300.0,
            fuel_cost_per_mwh=10.0,
        )
        # Need a generator in region_b for the model (0 output possible)
        g_b = make_generator(
            id="g_b", name="Gen B", region="region_b", capacity_mw=0.0,
            fuel_cost_per_mwh=100.0,
        )
        generators = [g_a, g_b]

        ic = make_interconnection(
            id="ic_ab", from_region="region_a", to_region="region_b", capacity_mw=200.0,
        )

        timesteps = [0]

        p: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for g in generators:
            for t in timesteps:
                p[(g.id, t)] = pulp.LpVariable(
                    f"p_{g.id}_{t}", lowBound=0, upBound=g.capacity_mw,
                )

        f: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for t in timesteps:
            f[(ic.id, t)] = pulp.LpVariable(f"f_{ic.id}_{t}", cat="Continuous")

        regional_demand = {
            "region_a": [100.0],
            "region_b": [80.0],
        }

        # Minimise total generation cost
        model += pulp.lpSum(
            g.fuel_cost_per_mwh * p[(g.id, t)] for g in generators for t in timesteps
        )

        add_nodal_balance_constraints(
            model, p, f, generators, [ic], timesteps, regional_demand,
        )
        add_transmission_capacity_constraints(model, f, [ic], timesteps)

        model.solve(pulp.PULP_CBC_CMD(msg=0))
        assert pulp.LpStatus[model.status] == "Optimal"

        flow_val = pulp.value(f[("ic_ab", 0)])
        # Region B has no generation, so flow must equal region B demand
        assert abs(flow_val - 80.0) < 1e-3, (
            f"Expected flow of 80 MW to cover region_b deficit, got {flow_val}"
        )


# ======================================================================
# TestPackageExports — public API
# ======================================================================


class TestPackageExports:
    """Tests that Interconnection and InterconnectionFlow are exported from src.uc."""

    def test_interconnection_importable_from_uc(self) -> None:
        """Interconnection is importable from the src.uc package."""
        from src.uc import Interconnection as IC

        ic = IC(
            id="test",
            name_en="Test",
            from_region="a",
            to_region="b",
            capacity_mw=100.0,
        )
        assert ic.id == "test"

    def test_interconnection_flow_importable_from_uc(self) -> None:
        """InterconnectionFlow is importable from the src.uc package."""
        from src.uc import InterconnectionFlow as ICFlow

        flow = ICFlow(interconnection_id="test", flow_mw=[1.0, 2.0])
        assert flow.interconnection_id == "test"


# ======================================================================
# Helpers — integration tests
# ======================================================================


def _flat_demand(mw: float, periods: int) -> DemandProfile:
    """Create a constant demand profile."""
    return DemandProfile(demands=[mw] * periods)


# ======================================================================
# TestTwoRegionIntegration — solve_uc with interconnection
# ======================================================================


class TestTwoRegionIntegration:
    """Integration tests for solve_uc with interconnections."""

    def test_two_region_with_interconnection(self) -> None:
        """Two-region solve with interconnection: flow from cheap to expensive region."""
        # 2 cheap generators in tokyo (300 MW each, total 600 MW)
        g1 = make_generator(
            id="g_tokyo_1",
            name="Tokyo Gen 1",
            capacity_mw=300.0,
            region="tokyo",
            fuel_cost_per_mwh=10.0,
        )
        g2 = make_generator(
            id="g_tokyo_2",
            name="Tokyo Gen 2",
            capacity_mw=300.0,
            region="tokyo",
            fuel_cost_per_mwh=10.0,
        )
        # 1 expensive generator in chubu (200 MW)
        g3 = make_generator(
            id="g_chubu_1",
            name="Chubu Gen 1",
            capacity_mw=200.0,
            region="chubu",
            fuel_cost_per_mwh=100.0,
        )

        # Interconnection: tokyo -> chubu, 100 MW capacity
        ic = make_interconnection(
            id="ic_tokyo_chubu",
            from_region="tokyo",
            to_region="chubu",
            capacity_mw=100.0,
        )

        th = TimeHorizon(num_periods=4, period_duration_h=1.0)
        # Total demand 750 MW; proportional split by capacity (600:200):
        #   tokyo = 750 * 0.75 = 562.5 MW, chubu = 750 * 0.25 = 187.5 MW
        # Tokyo surplus (600 - 562.5 = 37.5 MW) at 10 $/MWh is cheaper
        # than chubu at 100 $/MWh, so optimizer exports from tokyo to chubu.
        dp = _flat_demand(750.0, 4)
        params = UCParameters(
            generators=[g1, g2, g3],
            demand=dp,
            time_horizon=th,
            interconnections=[ic],
        )

        result = solve_uc(params)

        # (1) Optimal status
        assert result.status == "Optimal"
        assert result.total_cost > 0

        # (2) Positive flow from tokyo to chubu
        assert len(result.interconnection_flows) == 1
        ic_flow = result.interconnection_flows[0]
        assert ic_flow.interconnection_id == "ic_tokyo_chubu"
        assert len(ic_flow.flow_mw) == 4
        for t, flow_val in enumerate(ic_flow.flow_mw):
            assert flow_val > 0, (
                f"Expected positive flow at t={t}, got {flow_val}"
            )

        # (3) Flow <= 100 MW capacity bound
        for t, flow_val in enumerate(ic_flow.flow_mw):
            assert flow_val <= 100.0 + 1e-3, (
                f"Flow at t={t} exceeds capacity: {flow_val}"
            )

        # (4) Both regional demands met
        # Proportional split: tokyo = 75%, chubu = 25%
        tokyo_demand = 750.0 * (600.0 / 800.0)  # 562.5
        chubu_demand = 750.0 * (200.0 / 800.0)  # 187.5
        for t in range(4):
            tokyo_gen = sum(
                s.power_output_mw[t]
                for s in result.schedules
                if s.generator_id.startswith("g_tokyo")
            )
            chubu_gen = sum(
                s.power_output_mw[t]
                for s in result.schedules
                if s.generator_id.startswith("g_chubu")
            )
            flow = ic_flow.flow_mw[t]
            # tokyo: generation - export >= demand
            assert tokyo_gen - flow >= tokyo_demand - 1e-3, (
                f"Tokyo demand not met at t={t}: gen={tokyo_gen}, "
                f"export={flow}, demand={tokyo_demand}"
            )
            # chubu: generation + import >= demand
            assert chubu_gen + flow >= chubu_demand - 1e-3, (
                f"Chubu demand not met at t={t}: gen={chubu_gen}, "
                f"import={flow}, demand={chubu_demand}"
            )


# ======================================================================
# TestSolveWithoutInterconnections — regression
# ======================================================================


class TestSolveWithoutInterconnections:
    """Regression tests: solve_uc without interconnections is unchanged."""

    def test_solve_without_interconnections_unchanged(self) -> None:
        """solve_uc with no interconnections produces standard optimal result."""
        # 3-generator setup following test_uc_solver.py pattern
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

        th = TimeHorizon(num_periods=24, period_duration_h=1.0)
        dp = _flat_demand(200.0, 24)
        params = UCParameters(
            generators=[g1, g2, g3],
            demand=dp,
            time_horizon=th,
            interconnections=[],  # No interconnections
        )

        result = solve_uc(params)

        # Status and basic structure
        assert result.status == "Optimal"
        assert result.total_cost > 0
        assert result.solve_time_s >= 0
        assert len(result.schedules) == 3

        # No interconnection flows in result
        assert len(result.interconnection_flows) == 0

        # All generators have schedules
        gen_ids = {s.generator_id for s in result.schedules}
        assert gen_ids == {"g1", "g2", "g3"}

        # Demand balance holds at every timestep
        for t in range(24):
            total_gen = sum(s.power_output_mw[t] for s in result.schedules)
            assert total_gen >= 200.0 - 1e-3, (
                f"Demand balance violated at t={t}: "
                f"generation={total_gen:.4f} < demand=200.0"
            )

        # Total cost equals sum of generator costs
        sum_gen_costs = sum(s.total_cost for s in result.schedules)
        assert abs(result.total_cost - sum_gen_costs) < 1.0


# ======================================================================
# Helpers — decomposition tests
# ======================================================================


def _make_two_region_generators() -> List[Generator]:
    """Create generators across two regions for decomposition tests.

    Returns 4 generators in 2 regions (total 500 MW):
    - tokyo: g_tokyo_1 (200 MW coal), g_tokyo_2 (100 MW lng)
    - chubu: g_chubu_1 (150 MW coal), g_chubu_2 (50 MW oil)
    """
    g1 = make_generator(
        id="g_tokyo_1",
        name="Tokyo Coal",
        capacity_mw=200.0,
        fuel_type="coal",
        region="tokyo",
        p_min_mw=50.0,
        fuel_cost_per_mwh=30.0,
        no_load_cost=100.0,
        startup_cost=5000.0,
    )
    g2 = make_generator(
        id="g_tokyo_2",
        name="Tokyo LNG",
        capacity_mw=100.0,
        fuel_type="lng",
        region="tokyo",
        p_min_mw=20.0,
        fuel_cost_per_mwh=50.0,
        no_load_cost=50.0,
        startup_cost=2000.0,
    )
    g3 = make_generator(
        id="g_chubu_1",
        name="Chubu Coal",
        capacity_mw=150.0,
        fuel_type="coal",
        region="chubu",
        p_min_mw=30.0,
        fuel_cost_per_mwh=35.0,
        no_load_cost=80.0,
        startup_cost=4000.0,
    )
    g4 = make_generator(
        id="g_chubu_2",
        name="Chubu Oil",
        capacity_mw=50.0,
        fuel_type="oil",
        region="chubu",
        p_min_mw=10.0,
        fuel_cost_per_mwh=80.0,
        no_load_cost=20.0,
        startup_cost=1000.0,
    )
    return [g1, g2, g3, g4]


# ======================================================================
# TestDecompositionFallback — RegionalDecomposer with interconnections
# ======================================================================


class TestDecompositionFallback:
    """Tests that RegionalDecomposer falls back to national MILP with interconnections."""

    def test_decomposition_fallback_with_interconnections(self) -> None:
        """RegionalDecomposer returns [params] when interconnections are present."""
        gens = _make_two_region_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)

        ic = make_interconnection(
            id="ic_tc",
            from_region="tokyo",
            to_region="chubu",
            capacity_mw=100.0,
        )

        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            interconnections=[ic],
        )

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)

        # Should return single partition (no decomposition)
        assert len(partitions) == 1
        assert partitions[0] is params

    def test_decomposition_fallback_preserves_all_generators(self) -> None:
        """Fallback partition contains all original generators."""
        gens = _make_two_region_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)

        ic = make_interconnection(
            id="ic_tc",
            from_region="tokyo",
            to_region="chubu",
            capacity_mw=100.0,
        )

        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            interconnections=[ic],
        )

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)

        gen_ids = {g.id for g in partitions[0].generators}
        expected_ids = {g.id for g in gens}
        assert gen_ids == expected_ids

    def test_decomposition_fallback_preserves_interconnections(self) -> None:
        """Fallback partition preserves the interconnections list."""
        gens = _make_two_region_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)

        ic1 = make_interconnection(
            id="ic_01", from_region="tokyo", to_region="chubu", capacity_mw=100.0,
        )
        ic2 = make_interconnection(
            id="ic_02", from_region="chubu", to_region="tokyo", capacity_mw=200.0,
        )

        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            interconnections=[ic1, ic2],
        )

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)

        assert len(partitions[0].interconnections) == 2
        ic_ids = {ic.id for ic in partitions[0].interconnections}
        assert ic_ids == {"ic_01", "ic_02"}

    def test_decomposition_without_interconnections_unchanged(self) -> None:
        """RegionalDecomposer decomposes normally when interconnections list is empty."""
        gens = _make_two_region_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)

        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            interconnections=[],  # No interconnections
        )

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)

        # Should decompose into 2 regions (tokyo + chubu)
        assert len(partitions) == 2

        # Verify each partition has generators from only one region
        for p in partitions:
            regions = {g.region for g in p.generators}
            assert len(regions) == 1, (
                f"Partition has generators from multiple regions: {regions}"
            )

    def test_decomposition_without_interconnections_field_decomposes(self) -> None:
        """RegionalDecomposer decomposes when interconnections defaults (empty)."""
        gens = _make_two_region_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)

        # Don't pass interconnections — uses default empty list
        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
        )

        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)

        # Should decompose into 2 regions
        assert len(partitions) == 2


# ======================================================================
# TestConfigToggle — interconnection.enabled config flag
# ======================================================================


class TestConfigToggle:
    """Tests for the interconnection.enabled configuration toggle."""

    def test_config_toggle_disabled(self, project_root: Path) -> None:
        """With interconnection.enabled=false, interconnections are not loaded/used."""
        config_path = project_root / "config" / "uc_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        ic_config = config.get("interconnection", {})

        # Default config has enabled=false
        assert ic_config.get("enabled") is False, (
            "Default config should have interconnection.enabled=false"
        )

        # When disabled, a solve should NOT use interconnections
        # Create a simple 2-region problem
        gens = _make_two_region_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)

        # Simulate config-driven behavior: if not enabled, don't add interconnections
        if not ic_config.get("enabled", False):
            params = UCParameters(
                generators=gens,
                demand=dp,
                time_horizon=th,
                interconnections=[],  # Disabled → no interconnections
            )
        else:
            # This branch would load from data_path
            loader = InterconnectionLoader()
            ics = loader.load(str(project_root / ic_config["data_path"]))
            params = UCParameters(
                generators=gens,
                demand=dp,
                time_horizon=th,
                interconnections=ics,
            )

        result = solve_uc(params)

        assert result.status == "Optimal"
        # With disabled config, no interconnection flows should be present
        assert len(result.interconnection_flows) == 0

    def test_config_has_data_path(self, project_root: Path) -> None:
        """Config specifies a valid data_path for interconnections YAML."""
        config_path = project_root / "config" / "uc_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        ic_config = config.get("interconnection", {})
        data_path = ic_config.get("data_path")
        assert data_path is not None, "interconnection.data_path should be set"
        assert data_path == "data/reference/interconnections.yaml"

        # Verify the file actually exists
        full_path = project_root / data_path
        assert full_path.exists(), f"Interconnection data file not found: {full_path}"

    def test_config_toggle_enabled_loads_interconnections(
        self, project_root: Path
    ) -> None:
        """When enabled=true would be set, interconnections can be loaded from data_path."""
        config_path = project_root / "config" / "uc_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        ic_config = config.get("interconnection", {})
        data_path = ic_config.get("data_path")

        # Simulate enabled=true: load interconnections from data_path
        loader = InterconnectionLoader()
        ics = loader.load(str(project_root / data_path))

        assert len(ics) == 9
        assert all(isinstance(ic, Interconnection) for ic in ics)


# ======================================================================
# TestRegionalDecomposerSolveWithInterconnections — solve_decomposed
# ======================================================================


class TestRegionalDecomposerSolveWithInterconnections:
    """Tests that solve_decomposed() falls back to national MILP with interconnections."""

    def test_regional_decomposer_solve_with_interconnections(self) -> None:
        """solve_decomposed() falls back to national MILP and produces Optimal result."""
        gens = _make_two_region_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        # Low demand so the problem is easily feasible
        dp = _flat_demand(200.0, 6)

        ic = make_interconnection(
            id="ic_tc",
            from_region="tokyo",
            to_region="chubu",
            capacity_mw=100.0,
        )

        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            interconnections=[ic],
        )

        decomposer = RegionalDecomposer()
        result = decomposer.solve_decomposed(params)

        assert result.status == "Optimal"
        assert result.total_cost > 0
        assert result.solve_time_s >= 0

    def test_solve_decomposed_has_all_generators(self) -> None:
        """Merged result from solve_decomposed contains all generators."""
        gens = _make_two_region_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)

        ic = make_interconnection(
            id="ic_tc",
            from_region="tokyo",
            to_region="chubu",
            capacity_mw=100.0,
        )

        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            interconnections=[ic],
        )

        decomposer = RegionalDecomposer()
        result = decomposer.solve_decomposed(params)

        gen_ids_in_result = {s.generator_id for s in result.schedules}
        expected_ids = {g.id for g in gens}
        assert gen_ids_in_result == expected_ids, (
            f"Missing: {expected_ids - gen_ids_in_result}, "
            f"Extra: {gen_ids_in_result - expected_ids}"
        )

    def test_solve_decomposed_single_partition_solves_with_interconnections(
        self,
    ) -> None:
        """Single partition from fallback produces correct solve_uc result with flows."""
        gens = _make_two_region_generators()
        th = TimeHorizon(num_periods=6, period_duration_h=1.0)
        dp = _flat_demand(200.0, 6)

        ic = make_interconnection(
            id="ic_tc",
            from_region="tokyo",
            to_region="chubu",
            capacity_mw=100.0,
        )

        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            interconnections=[ic],
        )

        # Verify the partition returns [params] and solve_uc on that
        # partition produces interconnection flows correctly.
        decomposer = RegionalDecomposer()
        partitions = decomposer.partition(params)
        assert len(partitions) == 1

        # Solve the single partition directly (as solve_decomposed does)
        direct_result = solve_uc(partitions[0])
        assert direct_result.status == "Optimal"
        assert len(direct_result.interconnection_flows) == 1
        assert direct_result.interconnection_flows[0].interconnection_id == "ic_tc"
        assert len(direct_result.interconnection_flows[0].flow_mw) == 6

    def test_solve_decomposed_schedule_lengths(self) -> None:
        """All schedules from solve_decomposed have correct length."""
        gens = _make_two_region_generators()
        num_periods = 6
        th = TimeHorizon(num_periods=num_periods, period_duration_h=1.0)
        dp = _flat_demand(200.0, num_periods)

        ic = make_interconnection(
            id="ic_tc",
            from_region="tokyo",
            to_region="chubu",
            capacity_mw=100.0,
        )

        params = UCParameters(
            generators=gens,
            demand=dp,
            time_horizon=th,
            interconnections=[ic],
        )

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
