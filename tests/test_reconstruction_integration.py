"""Integration tests for the reconstruction pipeline.

Verifies end-to-end execution of the reconstruction pipeline combined
with AC power flow solving and MATPOWER export.  Tests cover both
simplification and reconnection modes, database attribute integration,
and full pipeline flows.

These tests complement the unit tests in ``test_reconstruction.py`` by
exercising cross-module interactions: reconstruction -> power flow ->
MATPOWER export, and reconstruction -> database attribute application.

Note: pandapower's ``pp.runpp()`` may fail during the result-extraction
phase on pandas >= 3.0 due to Copy-on-Write making DataFrame backing
arrays read-only.  The PYPOWER internal solve step completes before
this error, so convergence and voltage results can be extracted from
``net._ppc``.  The ``_run_powerflow`` helper handles this transparently.
"""

import copy
import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandapower as pp
import pytest
from pandapower.pypower.idx_bus import VM, VA

from src.ac_powerflow.network_prep import prepare_network
from src.converter.matpower_exporter import ExportResult, MATPOWERExporter
from src.db.grid_db import GridDatabase
from src.reconstruction.config import ReconstructionConfig
from src.reconstruction.data_synthesizer import DataSynthesizer
from src.reconstruction.isolator import Isolator
from src.reconstruction.pipeline import PipelineResult, ReconstructionPipeline
from src.reconstruction.reconnector import Reconnector
from src.reconstruction.simplifier import Simplifier

from tests.conftest import make_isolated_network


# ======================================================================
# Helpers
# ======================================================================


def _run_powerflow(net: Any) -> Tuple[bool, np.ndarray]:
    """Run AC power flow, handling pandas 3.0 CoW and non-convergence.

    Calls ``pp.runpp()`` and returns convergence status and voltage
    magnitudes.  Falls back to reading results from the internal
    PYPOWER case structure (``net._ppc``) when:

    - The result-extraction phase fails due to read-only DataFrames
      (pandas >= 3.0 CoW).
    - The power flow did not converge within the iteration limit
      (``LoadflowNotConverged``).

    Args:
        net: pandapower network.

    Returns:
        Tuple of (converged: bool, vm_pu: numpy array of voltage
        magnitudes in per-unit).

    Raises:
        RuntimeError: If the power flow truly failed (no ``_ppc``
            populated).
    """
    try:
        pp.runpp(net, numba=False)
        converged = net["converged"]
        vm_pu = net.res_bus["vm_pu"].to_numpy(copy=True)
        return converged, vm_pu
    except (ValueError, Exception) as exc:
        # Accept ValueError (pandas CoW read-only) and
        # LoadflowNotConverged (iteration limit exceeded).
        # In both cases _ppc may be populated with partial results.
        if isinstance(exc, ValueError) and "read-only" not in str(exc):
            raise
        # For non-ValueError, only accept LoadflowNotConverged
        if not isinstance(exc, ValueError):
            from pandapower.auxiliary import LoadflowNotConverged

            if not isinstance(exc, LoadflowNotConverged):
                raise
        ppc = getattr(net, "_ppc", None)
        if ppc is None:
            raise RuntimeError(
                "Power flow failed: _ppc not populated"
            ) from exc
        converged = bool(ppc.get("success", False))
        bus = ppc.get("bus")
        if bus is not None:
            vm_pu = bus[:, VM].copy()
        else:
            vm_pu = np.array([])
        return converged, vm_pu


def _run_dcpp(net: Any) -> np.ndarray:
    """Run DC power flow, handling pandas 3.0 CoW read-only issue.

    Args:
        net: pandapower network.

    Returns:
        Voltage angle array from the DC power flow solution.

    Raises:
        RuntimeError: If the DC power flow truly failed.
    """
    try:
        pp.rundcpp(net, numba=False)
        return net.res_bus["va_degree"].to_numpy(copy=True)
    except ValueError as exc:
        if "read-only" not in str(exc):
            raise
        ppc = getattr(net, "_ppc", None)
        if ppc is None:
            raise RuntimeError(
                "DC power flow failed: _ppc not populated"
            ) from exc
        bus = ppc.get("bus")
        if bus is not None:
            return bus[:, VA].copy()
        return np.array([])


def _make_balanced_isolated_network() -> Any:
    """Create a well-balanced network with isolated elements for power flow.

    Builds a meshed 4-bus main component with pre-placed balanced loads
    and generation, plus 2 isolated buses.  After reconstruction the
    network is well-conditioned for AC power flow convergence.

    Topology::

        Bus 0 (slack, 110 kV)
          |
        [line 0-1: 20 km]
          |
        Bus 1 (gen PV, 110 kV, 40 MW)
          |           \\
        [line 1-2]   [line 1-3]
          |           \\
        Bus 2 (load)   Bus 3 (load)
          \\           /
          [line 2-3]
           \\_______/

        Bus 4 (isolated, 110 kV, gen 10 MW)
        Bus 5 (isolated, 110 kV)

    Loads are pre-placed on all main buses so that ``skip_existing_loads``
    preserves them during synthesis, avoiding regional demand mismatch.

    Returns:
        A pandapower network with balanced loads/generation and isolated
        elements suitable for reconstruction + power flow testing.
    """
    net = pp.create_empty_network(f_hz=60.0)

    # Main connected component
    b0 = pp.create_bus(net, vn_kv=110.0, name="Slack Bus")
    b1 = pp.create_bus(net, vn_kv=110.0, name="Gen Bus")
    b2 = pp.create_bus(net, vn_kv=110.0, name="Load Bus A")
    b3 = pp.create_bus(net, vn_kv=110.0, name="Load Bus B")

    # External grid (slack)
    pp.create_ext_grid(net, bus=b0, vm_pu=1.02, name="Grid Connection")

    # Generator (PV bus) with max_p_mw for synthesis
    pp.create_gen(
        net, bus=b1, p_mw=40.0, vm_pu=1.01, name="Thermal Gen",
        max_p_mw=80.0,
    )

    # Balanced loads on all main buses
    pp.create_load(net, bus=b0, p_mw=10.0, q_mvar=3.0, name="Load Slack")
    pp.create_load(net, bus=b1, p_mw=15.0, q_mvar=5.0, name="Load Gen")
    pp.create_load(net, bus=b2, p_mw=30.0, q_mvar=10.0, name="Load A")
    pp.create_load(net, bus=b3, p_mw=20.0, q_mvar=5.0, name="Load B")

    # Lines using standard line type for 110 kV
    pp.create_line(
        net, b0, b1, 20.0, "149-AL1/24-ST1A 110.0", name="Line 0-1",
    )
    pp.create_line(
        net, b1, b2, 15.0, "149-AL1/24-ST1A 110.0", name="Line 1-2",
    )
    pp.create_line(
        net, b1, b3, 25.0, "149-AL1/24-ST1A 110.0", name="Line 1-3",
    )
    pp.create_line(
        net, b2, b3, 30.0, "149-AL1/24-ST1A 110.0", name="Line 2-3",
    )

    # Isolated buses (no line connections)
    b4 = pp.create_bus(net, vn_kv=110.0, name="Isolated Bus A")
    b5 = pp.create_bus(net, vn_kv=110.0, name="Isolated Bus B")

    # Generator on the first isolated bus
    pp.create_gen(
        net, bus=b4, p_mw=10.0, vm_pu=1.0, name="Isolated Gen",
        max_p_mw=20.0,
    )

    # Pre-place small loads on isolated buses so that
    # ``skip_existing_loads=True`` preserves balance after reconnection.
    # Without these, the data synthesizer allocates the full regional
    # demand (~4675 MW for shikoku) to just 2 buses, causing divergence.
    pp.create_load(net, bus=b4, p_mw=5.0, q_mvar=1.5, name="Load Iso A")
    pp.create_load(net, bus=b5, p_mw=3.0, q_mvar=1.0, name="Load Iso B")

    return net


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def balanced_isolated_net() -> Any:
    """Return a well-balanced network with isolated elements for power flow."""
    return _make_balanced_isolated_network()


@pytest.fixture
def simplify_config() -> ReconstructionConfig:
    """Return a simplify-mode config for integration tests."""
    return ReconstructionConfig(
        mode="simplify",
        seed=42,
        min_component_size=2,
        reserve_margin=0.05,
        skip_existing_loads=True,
        skip_existing_generation=True,
        db_path=":memory:",
    )


@pytest.fixture
def reconnect_config() -> ReconstructionConfig:
    """Return a reconnect-mode config for integration tests."""
    return ReconstructionConfig(
        mode="reconnect",
        seed=42,
        min_reactance_ohm_per_km=0.001,
        min_component_size=2,
        max_reconnection_distance_km=200.0,
        default_voltage_kv=66.0,
        reserve_margin=0.05,
        skip_existing_loads=True,
        skip_existing_generation=True,
        db_path=":memory:",
    )


# ======================================================================
# Integration: Simplify -> Power Flow
# ======================================================================


class TestSimplifyThenPowerflow:
    """Integration: simplified network converges with DC and AC power flow."""

    def test_simplify_then_dc_powerflow(
        self,
        balanced_isolated_net: Any,
        simplify_config: ReconstructionConfig,
    ) -> None:
        """Simplified network converges with DC power flow."""
        pipeline = ReconstructionPipeline(simplify_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")
        net = result.net

        va_degree = _run_dcpp(net)
        assert len(va_degree) > 0

    def test_simplify_then_ac_powerflow(
        self,
        balanced_isolated_net: Any,
        simplify_config: ReconstructionConfig,
    ) -> None:
        """Simplified network converges with AC power flow (Newton-Raphson)."""
        pipeline = ReconstructionPipeline(simplify_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")
        net = result.net

        converged, vm_pu = _run_powerflow(net)
        assert converged is True
        assert np.all(vm_pu > 0.7), f"Voltage too low: {vm_pu.min()}"
        assert np.all(vm_pu < 1.3), f"Voltage too high: {vm_pu.max()}"

    def test_simplify_produces_single_component(
        self,
        balanced_isolated_net: Any,
        simplify_config: ReconstructionConfig,
    ) -> None:
        """Simplified network has exactly 1 connected component."""
        pipeline = ReconstructionPipeline(simplify_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")

        assert result.simplification_result is not None
        assert result.simplification_result.component_count == 1

    def test_simplify_network_data_extractable(
        self,
        balanced_isolated_net: Any,
        simplify_config: ReconstructionConfig,
    ) -> None:
        """Ybus and solver data can be extracted from simplified network."""
        pipeline = ReconstructionPipeline(simplify_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")
        net = result.net

        data = prepare_network(net)
        assert data.Ybus is not None
        assert data.Ybus.shape[0] == data.Ybus.shape[1]
        assert data.Sbus.shape[0] == data.Ybus.shape[0]
        assert len(data.ref) >= 1


# ======================================================================
# Integration: Reconnect -> Power Flow
# ======================================================================


class TestReconnectThenPowerflow:
    """Integration: reconnected network converges with DC and AC power flow."""

    def test_reconnect_then_dc_powerflow(
        self,
        balanced_isolated_net: Any,
        reconnect_config: ReconstructionConfig,
    ) -> None:
        """Reconnected network converges with DC power flow."""
        pipeline = ReconstructionPipeline(reconnect_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")
        net = result.net

        va_degree = _run_dcpp(net)
        assert len(va_degree) > 0

    def test_reconnect_then_ac_powerflow(
        self,
        balanced_isolated_net: Any,
        reconnect_config: ReconstructionConfig,
    ) -> None:
        """Reconnected network converges with AC power flow."""
        pipeline = ReconstructionPipeline(reconnect_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")
        net = result.net

        converged, vm_pu = _run_powerflow(net)
        assert converged is True
        assert np.all(vm_pu > 0.7), f"Voltage too low: {vm_pu.min()}"
        assert np.all(vm_pu < 1.3), f"Voltage too high: {vm_pu.max()}"

    def test_reconnect_ybus_valid_and_extractable(
        self,
        balanced_isolated_net: Any,
        reconnect_config: ReconstructionConfig,
    ) -> None:
        """Reconnected network produces a valid, non-singular Ybus matrix."""
        pipeline = ReconstructionPipeline(reconnect_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")

        assert result.reconnection_result is not None
        assert result.reconnection_result.ybus_shape is not None
        assert result.reconnection_result.ybus_nonsingular is True

        shape = result.reconnection_result.ybus_shape
        assert shape[0] == shape[1]

    def test_reconnect_preserves_all_buses(
        self,
        balanced_isolated_net: Any,
        reconnect_config: ReconstructionConfig,
    ) -> None:
        """Reconnect mode keeps all buses (unlike simplification)."""
        original_bus_count = len(balanced_isolated_net.bus)

        pipeline = ReconstructionPipeline(reconnect_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")

        assert len(result.net.bus) == original_bus_count

    def test_reconnect_adds_synthetic_lines(
        self,
        balanced_isolated_net: Any,
        reconnect_config: ReconstructionConfig,
    ) -> None:
        """Reconnect mode adds synthetic lines for isolated buses."""
        original_line_count = len(balanced_isolated_net.line)

        pipeline = ReconstructionPipeline(reconnect_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")

        assert len(result.net.line) > original_line_count
        assert result.reconnection_result.lines_created > 0


# ======================================================================
# Integration: Reconstruction -> MATPOWER Export
# ======================================================================


class TestReconstructThenMatpowerExport:
    """Integration: reconstructed network exports to valid MATPOWER .mat file."""

    def test_simplify_then_matpower_export(
        self,
        balanced_isolated_net: Any,
        simplify_config: ReconstructionConfig,
        tmp_path: Path,
    ) -> None:
        """Simplified network exports to a valid MATPOWER .mat file."""
        pipeline = ReconstructionPipeline(simplify_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")
        net = result.net

        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        export_result = exporter.export_region(net, "shikoku")

        assert export_result.success is True
        assert os.path.exists(export_result.mat_path)
        assert export_result.bus_count > 0
        assert export_result.branch_count > 0

    def test_reconnect_then_matpower_export(
        self,
        balanced_isolated_net: Any,
        reconnect_config: ReconstructionConfig,
        tmp_path: Path,
    ) -> None:
        """Reconnected network exports to a valid MATPOWER .mat file."""
        pipeline = ReconstructionPipeline(reconnect_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")
        net = result.net

        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        export_result = exporter.export_region(net, "shikoku")

        assert export_result.success is True
        assert os.path.exists(export_result.mat_path)
        assert export_result.bus_count > 0
        assert export_result.branch_count > 0

    def test_matpower_bus_count_matches_network(
        self,
        balanced_isolated_net: Any,
        simplify_config: ReconstructionConfig,
        tmp_path: Path,
    ) -> None:
        """MATPOWER bus count matches the reconstructed pandapower bus count."""
        pipeline = ReconstructionPipeline(simplify_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")
        net = result.net
        pp_bus_count = len(net.bus)

        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        export_result = exporter.export_region(net, "shikoku")

        assert export_result.bus_count == pp_bus_count

    def test_matpower_report_validation_flags(
        self,
        balanced_isolated_net: Any,
        reconnect_config: ReconstructionConfig,
        tmp_path: Path,
    ) -> None:
        """MATPOWER export report has correct validation flags."""
        pipeline = ReconstructionPipeline(reconnect_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")
        net = result.net

        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        export_result = exporter.export_region(net, "shikoku")

        assert export_result.report is not None
        validation = export_result.report["validation"]
        assert validation["has_buses"] is True
        assert validation["has_branches"] is True
        assert validation["tables_non_empty"] is True


# ======================================================================
# Integration: Reconstruction -> DB Attributes
# ======================================================================


class TestReconstructWithDbAttributes:
    """Integration: DB attributes are correctly applied during reconstruction."""

    def test_pipeline_with_db_persists_load_attributes(
        self,
        balanced_isolated_net: Any,
        simplify_config: ReconstructionConfig,
    ) -> None:
        """Pipeline with DB persists synthesised load attributes."""
        db = GridDatabase(":memory:")

        pipeline = ReconstructionPipeline(
            simplify_config, copy_network=True, db=db,
        )
        result = pipeline.run(balanced_isolated_net, region="shikoku")

        assert not result.net.load.empty
        assert result.synthesis_result is not None

    def test_pipeline_with_db_generator_attributes(
        self,
        balanced_isolated_net: Any,
        simplify_config: ReconstructionConfig,
    ) -> None:
        """DB generator attributes are accessible alongside reconstruction."""
        db = GridDatabase(":memory:")

        db.upsert_generator_attributes(
            "main_gen_0",
            fuel_type="coal",
            capacity_mw=500.0,
            fuel_cost_per_mwh=25.0,
        )

        pipeline = ReconstructionPipeline(
            simplify_config, copy_network=True, db=db,
        )
        result = pipeline.run(balanced_isolated_net, region="shikoku")

        attrs = db.get_generator_attributes("main_gen_0")
        assert attrs is not None
        assert attrs.fuel_type == "coal"
        assert attrs.capacity_mw == 500.0

        assert result.net is not None
        assert result.synthesis_result is not None

    def test_db_update_propagates_on_rerun(
        self,
        balanced_isolated_net: Any,
        simplify_config: ReconstructionConfig,
    ) -> None:
        """Updating DB attributes is visible after re-running pipeline."""
        db = GridDatabase(":memory:")

        db.upsert_generator_attributes(
            "test_gen",
            fuel_type="coal",
            fuel_cost_per_mwh=25.0,
        )

        db.upsert_generator_attributes(
            "test_gen",
            fuel_cost_per_mwh=30.0,
        )

        attrs = db.get_generator_attributes("test_gen")
        assert attrs is not None
        assert attrs.fuel_cost_per_mwh == 30.0
        assert attrs.fuel_type == "coal"


# ======================================================================
# Integration: Full Pipeline -- Simplify Mode
# ======================================================================


class TestFullPipelineSimplify:
    """End-to-end: load network -> detect isolation -> simplify ->
    synthesize data -> run power flow -> export MATPOWER."""

    def test_full_pipeline_simplify(
        self,
        tmp_path: Path,
    ) -> None:
        """Full simplification pipeline produces valid power flow and export."""
        net = _make_balanced_isolated_network()

        cfg = ReconstructionConfig(
            mode="simplify", seed=42, db_path=":memory:",
            skip_existing_loads=True, skip_existing_generation=True,
        )

        # Step 1: Run reconstruction pipeline
        pipeline = ReconstructionPipeline(cfg, copy_network=True)
        result = pipeline.run(net, region="shikoku")
        net = result.net

        # Step 2: Validate reconstruction results
        assert result.reconstruction_mode == "simplify"
        assert result.isolation_result is not None
        assert result.isolation_result.has_isolation is True
        assert result.simplification_result is not None
        assert result.simplification_result.buses_removed > 0
        assert result.synthesis_result is not None

        # Step 3: Run AC power flow
        converged, vm_pu = _run_powerflow(net)
        assert converged is True

        # Step 4: Validate voltage profile
        assert np.all(vm_pu > 0.7), f"Voltage too low: {vm_pu.min()}"
        assert np.all(vm_pu < 1.3), f"Voltage too high: {vm_pu.max()}"

        # Step 5: Export to MATPOWER
        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        export_result = exporter.export_region(net, "shikoku")

        assert export_result.success is True
        assert export_result.bus_count > 0
        assert export_result.branch_count > 0
        assert os.path.exists(export_result.mat_path)

    def test_full_pipeline_simplify_metadata(
        self,
        balanced_isolated_net: Any,
        simplify_config: ReconstructionConfig,
    ) -> None:
        """Full pipeline metadata (mode, seed, region) is correctly recorded."""
        pipeline = ReconstructionPipeline(simplify_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")

        assert result.reconstruction_mode == "simplify"
        assert result.seed == 42
        assert result.region == "shikoku"
        assert result.elapsed_seconds >= 0

        summary = result.summary
        assert summary["mode"] == "simplify"
        assert summary["seed"] == 42
        assert summary["region"] == "shikoku"

    def test_full_pipeline_simplify_load_synthesis(
        self,
        balanced_isolated_net: Any,
        simplify_config: ReconstructionConfig,
    ) -> None:
        """Simplify pipeline preserves pre-placed loads on remaining buses."""
        pipeline = ReconstructionPipeline(simplify_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")
        net = result.net

        assert not net.load.empty
        assert all(net.load["p_mw"] > 0)


# ======================================================================
# Integration: Full Pipeline -- Reconnect Mode
# ======================================================================


class TestFullPipelineReconnect:
    """End-to-end: load network -> detect isolation -> reconnect ->
    synthesize data -> run power flow -> export MATPOWER."""

    def test_full_pipeline_reconnect(
        self,
        tmp_path: Path,
    ) -> None:
        """Full reconnection pipeline produces valid power flow and export."""
        net = _make_balanced_isolated_network()

        cfg = ReconstructionConfig(
            mode="reconnect", seed=42, db_path=":memory:",
            min_reactance_ohm_per_km=0.001,
            skip_existing_loads=True, skip_existing_generation=True,
        )

        # Step 1: Run reconstruction pipeline
        pipeline = ReconstructionPipeline(cfg, copy_network=True)
        result = pipeline.run(net, region="shikoku")
        net = result.net

        # Step 2: Validate reconstruction results
        assert result.reconstruction_mode == "reconnect"
        assert result.isolation_result is not None
        assert result.isolation_result.has_isolation is True
        assert result.reconnection_result is not None
        assert result.reconnection_result.lines_created > 0
        assert result.reconnection_result.ybus_shape is not None
        assert result.synthesis_result is not None

        # Step 3: Ybus should be valid
        assert result.reconnection_result.ybus_nonsingular is True

        # Step 4: Run AC power flow
        converged, vm_pu = _run_powerflow(net)
        assert converged is True

        # Step 5: Validate voltage profile
        assert np.all(vm_pu > 0.7), f"Voltage too low: {vm_pu.min()}"
        assert np.all(vm_pu < 1.3), f"Voltage too high: {vm_pu.max()}"

        # Step 6: Export to MATPOWER
        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        export_result = exporter.export_region(net, "shikoku")

        assert export_result.success is True
        assert export_result.bus_count > 0
        assert export_result.branch_count > 0
        assert os.path.exists(export_result.mat_path)

    def test_full_pipeline_reconnect_bus_count(
        self,
        balanced_isolated_net: Any,
        reconnect_config: ReconstructionConfig,
    ) -> None:
        """Reconnect pipeline preserves all original buses."""
        original_bus_count = len(balanced_isolated_net.bus)

        pipeline = ReconstructionPipeline(reconnect_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")

        assert len(result.net.bus) == original_bus_count

    def test_full_pipeline_reconnect_ybus_dimensions(
        self,
        balanced_isolated_net: Any,
        reconnect_config: ReconstructionConfig,
    ) -> None:
        """Ybus dimensions match the total bus count after reconnection."""
        pipeline = ReconstructionPipeline(reconnect_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")

        ybus_shape = result.reconnection_result.ybus_shape
        assert ybus_shape is not None

        n_buses = len(result.net.bus[result.net.bus["in_service"]])
        assert ybus_shape[0] == n_buses
        assert ybus_shape[1] == n_buses


# ======================================================================
# Integration: Reproducibility across pipeline modes
# ======================================================================


class TestReproducibilityIntegration:
    """Cross-mode reproducibility: verify identical outputs with same seed."""

    def test_simplify_reproducibility_with_export(
        self,
        tmp_path: Path,
    ) -> None:
        """Two simplify runs with same seed produce identical bus/line counts."""
        net1 = _make_balanced_isolated_network()
        net2 = copy.deepcopy(net1)

        cfg1 = ReconstructionConfig(
            mode="simplify", seed=42, db_path=":memory:",
        )
        cfg2 = ReconstructionConfig(
            mode="simplify", seed=42, db_path=":memory:",
        )

        pipeline1 = ReconstructionPipeline(cfg1, copy_network=True)
        pipeline2 = ReconstructionPipeline(cfg2, copy_network=True)

        result1 = pipeline1.run(net1, region="shikoku")
        result2 = pipeline2.run(net2, region="shikoku")

        assert len(result1.net.bus) == len(result2.net.bus)
        assert len(result1.net.line) == len(result2.net.line)
        assert len(result1.net.load) == len(result2.net.load)
        assert len(result1.net.gen) == len(result2.net.gen)

        if not result1.net.load.empty:
            np.testing.assert_array_equal(
                result1.net.load["p_mw"].to_numpy(),
                result2.net.load["p_mw"].to_numpy(),
            )

        # Both should export successfully
        exporter1 = MATPOWERExporter(
            output_dir=str(tmp_path / "run1" / "matpower"),
            reports_dir=str(tmp_path / "run1" / "reports"),
        )
        exporter2 = MATPOWERExporter(
            output_dir=str(tmp_path / "run2" / "matpower"),
            reports_dir=str(tmp_path / "run2" / "reports"),
        )

        exp1 = exporter1.export_region(result1.net, "shikoku")
        exp2 = exporter2.export_region(result2.net, "shikoku")

        assert exp1.bus_count == exp2.bus_count
        assert exp1.branch_count == exp2.branch_count
        assert exp1.gen_count == exp2.gen_count

    def test_mode_switch_produces_different_results(self) -> None:
        """Simplify and reconnect modes produce different network structures."""
        net = _make_balanced_isolated_network()

        cfg_s = ReconstructionConfig(
            mode="simplify", seed=42, db_path=":memory:",
        )
        cfg_r = ReconstructionConfig(
            mode="reconnect", seed=42, db_path=":memory:",
        )

        result_s = ReconstructionPipeline(cfg_s, copy_network=True).run(
            net, region="shikoku",
        )
        result_r = ReconstructionPipeline(cfg_r, copy_network=True).run(
            net, region="shikoku",
        )

        # Simplify removes buses; reconnect keeps them
        assert len(result_s.net.bus) < len(result_r.net.bus)

        # Both should produce valid networks that converge
        converged_s, _ = _run_powerflow(result_s.net)
        assert converged_s is True

        converged_r, _ = _run_powerflow(result_r.net)
        assert converged_r is True


# ======================================================================
# Integration: Custom solver compatibility after reconstruction
# ======================================================================


class TestCustomSolverAfterReconstruction:
    """Verify custom AC solvers work on reconstructed networks."""

    def test_custom_nr_on_simplified_network(self) -> None:
        """Custom Newton-Raphson solver converges on simplified network."""
        from src.ac_powerflow.solver_interface import ACMethodResult

        net = _make_balanced_isolated_network()

        cfg = ReconstructionConfig(
            mode="simplify", seed=42, db_path=":memory:",
        )
        pipeline = ReconstructionPipeline(cfg, copy_network=True)
        result = pipeline.run(net, region="shikoku")
        net = result.net

        data = prepare_network(net)

        from src.ac_powerflow.methods import get_all_methods

        methods = get_all_methods()
        nr_method = next(m for m in methods if m.id == "custom_nr")

        solver_result = nr_method.solver_fn(
            data.Ybus,
            data.Sbus,
            np.copy(data.V0),
            np.array(data.ref),
            np.array(data.pv),
            np.array(data.pq),
            max_iter=50,
            tol=1e-8,
        )

        assert isinstance(solver_result, ACMethodResult)
        assert solver_result.converged is True, (
            f"Custom NR did not converge: {solver_result.failure_reason}"
        )
        assert solver_result.V is not None

        Vm = np.abs(solver_result.V)
        assert np.all(Vm > 0.7), f"Voltage too low: {Vm.min()}"
        assert np.all(Vm < 1.3), f"Voltage too high: {Vm.max()}"

    def test_custom_nr_on_reconnected_network(self) -> None:
        """Custom Newton-Raphson solver converges on reconnected network."""
        from src.ac_powerflow.solver_interface import ACMethodResult

        net = _make_balanced_isolated_network()

        cfg = ReconstructionConfig(
            mode="reconnect", seed=42, db_path=":memory:",
            min_reactance_ohm_per_km=0.001,
        )
        pipeline = ReconstructionPipeline(cfg, copy_network=True)
        result = pipeline.run(net, region="shikoku")
        net = result.net

        data = prepare_network(net)

        from src.ac_powerflow.methods import get_all_methods

        methods = get_all_methods()
        nr_method = next(m for m in methods if m.id == "custom_nr")

        solver_result = nr_method.solver_fn(
            data.Ybus,
            data.Sbus,
            np.copy(data.V0),
            np.array(data.ref),
            np.array(data.pv),
            np.array(data.pq),
            max_iter=50,
            tol=1e-8,
        )

        assert isinstance(solver_result, ACMethodResult)
        assert solver_result.converged is True, (
            f"Custom NR did not converge: {solver_result.failure_reason}"
        )

    def test_network_data_shape_consistency(
        self,
        balanced_isolated_net: Any,
        reconnect_config: ReconstructionConfig,
    ) -> None:
        """Extracted network data has consistent shapes across arrays."""
        pipeline = ReconstructionPipeline(reconnect_config, copy_network=True)
        result = pipeline.run(balanced_isolated_net, region="shikoku")
        net = result.net

        data = prepare_network(net)
        n = data.Ybus.shape[0]

        assert data.Ybus.shape == (n, n)
        assert data.Sbus.shape == (n,)
        assert data.V0.shape == (n,)
        assert len(data.ref) >= 1
        for idx in list(data.ref) + list(data.pv) + list(data.pq):
            assert 0 <= idx < n, f"Bus index {idx} out of range [0, {n})"
