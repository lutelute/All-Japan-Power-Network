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
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pulp
import pytest

from src.model.generator import Generator
from src.uc.constraints import (
    add_nodal_balance_constraints,
    add_transmission_capacity_constraints,
)
from src.uc.interconnection_loader import InterconnectionLoader
from src.uc.models import Interconnection, InterconnectionFlow
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
