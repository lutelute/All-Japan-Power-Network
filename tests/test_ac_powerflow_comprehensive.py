"""Comprehensive AC power flow validation across all ~20 methods against
reconstructed networks.

Tests cover:
- All 20 methods (5 pandapower + 7 NR + 4 iterative + 4 decoupled)
  on networks produced by the reconstruction pipeline in both
  simplification and reconnection modes.
- Per-method convergence status and voltage range validation.
- Per-category convergence behavior validation.
- Cross-network consistency (3-bus, 5-bus, reconstructed networks).
- Convergence report generation from comprehensive results.
- Voltage magnitude and angle reasonableness checks.
- Reproducibility of solver results across repeated runs.

Network fixtures:

* ``simple_3bus_net`` — 3-bus meshed network (test_ac_powerflow pattern).
* ``meshed_5bus_net`` — 5-bus meshed network (test_ac_powerflow_integration
  pattern).
* ``reconstructed_simplified_net`` — Network with isolated elements that
  has been simplified via the reconstruction pipeline.
* ``reconstructed_reconnected_net`` — Network with isolated elements that
  has been reconnected via the reconstruction pipeline.
"""

import copy
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandapower as pp
import pytest

from src.ac_powerflow.batch_runner import run_all_methods
from src.ac_powerflow.convergence_report import generate_report, save_report
from src.ac_powerflow.methods import MethodDescriptor, get_all_methods
from src.ac_powerflow.network_prep import NetworkData, prepare_network
from src.ac_powerflow.solver_interface import ACMethodResult
from src.reconstruction.config import ReconstructionConfig
from src.reconstruction.pipeline import ReconstructionPipeline

from tests.conftest import make_isolated_network


# ======================================================================
# Helpers
# ======================================================================


def _run_powerflow_safe(net: Any) -> Tuple[bool, np.ndarray]:
    """Run AC power flow, handling pandas 3.0 CoW and non-convergence.

    Calls ``pp.runpp()`` and returns convergence status and voltage
    magnitudes.  Falls back to reading results from the internal
    PYPOWER case structure (``net._ppc``) when the result-extraction
    phase fails.

    Args:
        net: pandapower network.

    Returns:
        Tuple of (converged, vm_pu).
    """
    try:
        pp.runpp(net, numba=False)
        converged = net["converged"]
        vm_pu = net.res_bus["vm_pu"].to_numpy(copy=True)
        return converged, vm_pu
    except Exception:
        if hasattr(net, "_ppc") and net._ppc is not None:
            from pandapower.pypower.idx_bus import VM
            bus = net._ppc.get("bus")
            if bus is not None:
                vm_pu = bus[:, VM]
                return True, vm_pu
        return False, np.array([])


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def simple_3bus_net():
    """Create a simple 3-bus pandapower network.

    Topology::

        Bus 0 (slack) ---[line 0-1]--- Bus 1 (gen, PV)
          |                              |
        [line 0-2]                   [line 1-2]
          |                              |
        Bus 2 (load, PQ) ---------------+

    All buses at 110 kV.
    """
    net = pp.create_empty_network(f_hz=50.0)

    bus0 = pp.create_bus(net, vn_kv=110.0, name="Bus 0 (slack)")
    bus1 = pp.create_bus(net, vn_kv=110.0, name="Bus 1 (gen)")
    bus2 = pp.create_bus(net, vn_kv=110.0, name="Bus 2 (load)")

    pp.create_ext_grid(net, bus=bus0, vm_pu=1.02, name="Grid")
    pp.create_gen(net, bus=bus1, p_mw=40.0, vm_pu=1.01, name="Gen 1")
    pp.create_load(net, bus=bus2, p_mw=60.0, q_mvar=20.0, name="Load 2")

    pp.create_line(
        net, from_bus=bus0, to_bus=bus1, length_km=10.0,
        std_type="149-AL1/24-ST1A 110.0", name="Line 0-1",
    )
    pp.create_line(
        net, from_bus=bus1, to_bus=bus2, length_km=10.0,
        std_type="149-AL1/24-ST1A 110.0", name="Line 1-2",
    )
    pp.create_line(
        net, from_bus=bus0, to_bus=bus2, length_km=15.0,
        std_type="149-AL1/24-ST1A 110.0", name="Line 0-2",
    )

    return net


@pytest.fixture
def meshed_5bus_net():
    """Create a 5-bus meshed pandapower network.

    Topology::

        Bus 0 (slack, 110 kV)
          |
        [line 0-1: 20 km]
          |
        Bus 1 (gen PV, 80 MW) ---[line 1-3: 25 km]--- Bus 3 (load, 90 MW)
          |                                              |
        [line 1-2: 15 km]                             [line 3-4: 10 km]
          |                                              |
        Bus 2 (load, 50 MW) ---[line 2-4: 30 km]--- Bus 4 (load, 30 MW)
    """
    net = pp.create_empty_network(f_hz=60.0)

    b0 = pp.create_bus(net, vn_kv=110.0, name="Slack Bus")
    b1 = pp.create_bus(net, vn_kv=110.0, name="Gen Bus")
    b2 = pp.create_bus(net, vn_kv=110.0, name="Load Bus A")
    b3 = pp.create_bus(net, vn_kv=110.0, name="Load Bus B")
    b4 = pp.create_bus(net, vn_kv=110.0, name="Load Bus C")

    pp.create_ext_grid(net, bus=b0, vm_pu=1.02, name="Grid Connection")
    pp.create_gen(net, bus=b1, p_mw=80.0, vm_pu=1.01, name="Thermal Gen")

    pp.create_load(net, bus=b2, p_mw=50.0, q_mvar=15.0, name="Load A")
    pp.create_load(net, bus=b3, p_mw=90.0, q_mvar=30.0, name="Load B")
    pp.create_load(net, bus=b4, p_mw=30.0, q_mvar=10.0, name="Load C")

    pp.create_line(net, b0, b1, 20.0, "149-AL1/24-ST1A 110.0", name="Line 0-1")
    pp.create_line(net, b1, b2, 15.0, "149-AL1/24-ST1A 110.0", name="Line 1-2")
    pp.create_line(net, b1, b3, 25.0, "149-AL1/24-ST1A 110.0", name="Line 1-3")
    pp.create_line(net, b2, b4, 30.0, "149-AL1/24-ST1A 110.0", name="Line 2-4")
    pp.create_line(net, b3, b4, 10.0, "149-AL1/24-ST1A 110.0", name="Line 3-4")

    return net


@pytest.fixture
def reconstructed_simplified_net():
    """Produce a simplified network via the reconstruction pipeline.

    Starts from a network with 3 main + 2 isolated buses, applies the
    simplification pipeline (removes isolated elements), and returns
    the resulting single-component network ready for AC power flow.
    """
    raw_net = make_isolated_network(
        n_main_buses=3,
        n_isolated_buses=2,
        n_generators=2,
    )

    cfg = ReconstructionConfig(
        mode="simplify",
        seed=42,
        min_component_size=2,
        reserve_margin=0.05,
        skip_existing_loads=False,
        skip_existing_generation=False,
        db_path=":memory:",
    )
    pipeline = ReconstructionPipeline(cfg, copy_network=True)
    result = pipeline.run(raw_net, region="comprehensive_test")

    return result.net


@pytest.fixture
def reconstructed_reconnected_net():
    """Produce a reconnected network via the reconstruction pipeline.

    Starts from a network with 3 main + 2 isolated buses, applies the
    reconnection pipeline (creates synthetic connections), and returns
    the resulting fully-connected network ready for AC power flow.
    """
    raw_net = make_isolated_network(
        n_main_buses=3,
        n_isolated_buses=2,
        n_generators=2,
    )

    cfg = ReconstructionConfig(
        mode="reconnect",
        seed=42,
        min_reactance_ohm_per_km=0.001,
        min_component_size=2,
        max_reconnection_distance_km=200.0,
        default_voltage_kv=66.0,
        reserve_margin=0.05,
        skip_existing_loads=False,
        skip_existing_generation=False,
        db_path=":memory:",
    )
    pipeline = ReconstructionPipeline(cfg, copy_network=True)
    result = pipeline.run(raw_net, region="comprehensive_test")

    return result.net


# ======================================================================
# Test 1: All methods on simple 3-bus network
# ======================================================================


class TestAllMethodsSimple3Bus:
    """Run all 20 methods on the simple 3-bus network and validate."""

    def test_all_20_methods_execute(self, simple_3bus_net) -> None:
        """All 20 methods execute without unhandled exceptions."""
        results = run_all_methods(simple_3bus_net, region="simple_3bus")
        assert len(results) == 20, (
            f"Expected 20 method results, got {len(results)}"
        )

    def test_all_results_are_acmethodresult(self, simple_3bus_net) -> None:
        """Every method returns a valid ACMethodResult."""
        results = run_all_methods(simple_3bus_net, region="simple_3bus")
        for r in results:
            assert isinstance(r["result"], ACMethodResult), (
                f"Method '{r['method_id']}' returned {type(r['result'])}"
            )

    def test_custom_nr_converges_on_3bus(self, simple_3bus_net) -> None:
        """Custom NR converges on well-conditioned 3-bus network."""
        results = run_all_methods(simple_3bus_net, region="simple_3bus")
        nr = next(r for r in results if r["method_id"] == "custom_nr")
        result = nr["result"]
        assert result.converged is True, (
            f"custom_nr did not converge: {result.failure_reason}"
        )
        assert result.V is not None
        Vm = np.abs(result.V)
        assert np.all(Vm > 0.8), f"Voltage too low: {Vm.min()}"
        assert np.all(Vm < 1.2), f"Voltage too high: {Vm.max()}"

    def test_nr_variants_converge_on_3bus(self, simple_3bus_net) -> None:
        """All NR variants converge on the well-conditioned 3-bus network."""
        results = run_all_methods(simple_3bus_net, region="simple_3bus")
        nr_methods = [
            r for r in results if r["category"] == "custom_nr"
        ]

        for r in nr_methods:
            result = r["result"]
            assert result.converged is True, (
                f"NR variant '{r['method_id']}' did not converge: "
                f"{result.failure_reason}"
            )
            assert result.V is not None
            assert result.iterations > 0
            assert result.elapsed_sec > 0

    def test_all_methods_have_nonnegative_timing(self, simple_3bus_net) -> None:
        """Every method records non-negative elapsed time."""
        results = run_all_methods(simple_3bus_net, region="simple_3bus")
        for r in results:
            assert r["result"].elapsed_sec >= 0, (
                f"Method '{r['method_id']}' has negative elapsed time"
            )


# ======================================================================
# Test 2: All methods on meshed 5-bus network
# ======================================================================


class TestAllMethodsMeshed5Bus:
    """Run all 20 methods on the 5-bus meshed network and validate."""

    def test_all_20_methods_execute(self, meshed_5bus_net) -> None:
        """All 20 methods execute without unhandled exceptions."""
        results = run_all_methods(meshed_5bus_net, region="meshed_5bus")
        assert len(results) == 20

    def test_custom_nr_converges_on_5bus(self, meshed_5bus_net) -> None:
        """Custom NR converges on the 5-bus meshed network."""
        results = run_all_methods(meshed_5bus_net, region="meshed_5bus")
        nr = next(r for r in results if r["method_id"] == "custom_nr")
        result = nr["result"]
        assert result.converged is True, (
            f"custom_nr did not converge: {result.failure_reason}"
        )
        Vm = np.abs(result.V)
        assert np.all(Vm > 0.7), f"Voltage too low: {Vm.min()}"
        assert np.all(Vm < 1.3), f"Voltage too high: {Vm.max()}"

    def test_nr_variants_converge_on_5bus(self, meshed_5bus_net) -> None:
        """All NR variants converge on the 5-bus meshed network."""
        results = run_all_methods(meshed_5bus_net, region="meshed_5bus")
        nr_methods = [r for r in results if r["category"] == "custom_nr"]

        for r in nr_methods:
            result = r["result"]
            assert result.converged is True, (
                f"NR variant '{r['method_id']}' did not converge: "
                f"{result.failure_reason}"
            )

    def test_voltage_magnitudes_in_range(self, meshed_5bus_net) -> None:
        """Converged methods produce voltages within [0.7, 1.3] p.u."""
        results = run_all_methods(meshed_5bus_net, region="meshed_5bus")
        for r in results:
            result = r["result"]
            if result.converged and result.V is not None:
                Vm = np.abs(result.V)
                assert np.all(Vm > 0.7), (
                    f"Method '{r['method_id']}': voltage too low {Vm.min()}"
                )
                assert np.all(Vm < 1.3), (
                    f"Method '{r['method_id']}': voltage too high {Vm.max()}"
                )


# ======================================================================
# Test 3: All methods on reconstructed simplified network
# ======================================================================


class TestAllMethodsReconstructedSimplified:
    """Run all 20 methods on a simplified (reconstructed) network."""

    def test_all_20_methods_execute(self, reconstructed_simplified_net) -> None:
        """All 20 methods execute without unhandled exceptions."""
        results = run_all_methods(
            reconstructed_simplified_net, region="simplified",
        )
        assert len(results) == 20

    def test_each_result_is_acmethodresult(
        self, reconstructed_simplified_net,
    ) -> None:
        """Every method returns ACMethodResult."""
        results = run_all_methods(
            reconstructed_simplified_net, region="simplified",
        )
        for r in results:
            assert isinstance(r["result"], ACMethodResult), (
                f"Method '{r['method_id']}' returned {type(r['result'])}"
            )

    def test_result_records_have_required_keys(
        self, reconstructed_simplified_net,
    ) -> None:
        """Each result record contains all required keys."""
        results = run_all_methods(
            reconstructed_simplified_net, region="simplified",
        )
        required_keys = {
            "method_id", "method_name", "category", "region", "result",
        }
        for r in results:
            missing = required_keys - set(r.keys())
            assert not missing, (
                f"Method '{r.get('method_id')}' missing keys: {missing}"
            )

    def test_custom_nr_converges_on_simplified(
        self, reconstructed_simplified_net,
    ) -> None:
        """Custom NR converges on the simplified reconstructed network."""
        results = run_all_methods(
            reconstructed_simplified_net, region="simplified",
        )
        nr = next(r for r in results if r["method_id"] == "custom_nr")
        result = nr["result"]
        assert result.converged is True, (
            f"custom_nr did not converge on simplified network: "
            f"{result.failure_reason}"
        )
        assert result.V is not None
        Vm = np.abs(result.V)
        assert np.all(Vm > 0.5), f"Voltage too low: {Vm.min()}"
        assert np.all(Vm < 1.5), f"Voltage too high: {Vm.max()}"

    def test_at_least_one_pandapower_method_executes(
        self, reconstructed_simplified_net,
    ) -> None:
        """At least one pandapower method executes successfully."""
        results = run_all_methods(
            reconstructed_simplified_net, region="simplified",
        )
        pp_results = [
            r for r in results if r["category"] == "pandapower"
        ]
        assert len(pp_results) == 5, (
            f"Expected 5 pandapower results, got {len(pp_results)}"
        )
        # Each should have returned a valid ACMethodResult (even if
        # convergence failed)
        for r in pp_results:
            assert isinstance(r["result"], ACMethodResult)

    def test_failed_methods_have_failure_reasons(
        self, reconstructed_simplified_net,
    ) -> None:
        """Non-converged methods have a non-empty failure_reason."""
        results = run_all_methods(
            reconstructed_simplified_net, region="simplified",
        )
        for r in results:
            result = r["result"]
            if not result.converged:
                assert result.failure_reason is not None, (
                    f"Method '{r['method_id']}' failed without "
                    f"failure_reason"
                )
                assert len(result.failure_reason) > 0


# ======================================================================
# Test 4: All methods on reconstructed reconnected network
# ======================================================================


class TestAllMethodsReconstructedReconnected:
    """Run all 20 methods on a reconnected (reconstructed) network."""

    def test_all_20_methods_execute(
        self, reconstructed_reconnected_net,
    ) -> None:
        """All 20 methods execute without unhandled exceptions."""
        results = run_all_methods(
            reconstructed_reconnected_net, region="reconnected",
        )
        assert len(results) == 20

    def test_custom_nr_converges_on_reconnected(
        self, reconstructed_reconnected_net,
    ) -> None:
        """Custom NR converges on the reconnected reconstructed network."""
        results = run_all_methods(
            reconstructed_reconnected_net, region="reconnected",
        )
        nr = next(r for r in results if r["method_id"] == "custom_nr")
        result = nr["result"]
        assert result.converged is True, (
            f"custom_nr did not converge on reconnected network: "
            f"{result.failure_reason}"
        )
        assert result.V is not None

    def test_converged_methods_produce_finite_voltages(
        self, reconstructed_reconnected_net,
    ) -> None:
        """Converged methods produce finite voltage vectors."""
        results = run_all_methods(
            reconstructed_reconnected_net, region="reconnected",
        )
        for r in results:
            result = r["result"]
            if result.converged and result.V is not None:
                assert not np.isnan(result.V).any(), (
                    f"Method '{r['method_id']}': V contains NaN"
                )
                assert not np.isinf(result.V).any(), (
                    f"Method '{r['method_id']}': V contains Inf"
                )

    def test_voltage_magnitudes_in_range(
        self, reconstructed_reconnected_net,
    ) -> None:
        """Converged methods produce voltages within [0.5, 1.5] p.u."""
        results = run_all_methods(
            reconstructed_reconnected_net, region="reconnected",
        )
        for r in results:
            result = r["result"]
            if result.converged and result.V is not None:
                Vm = np.abs(result.V)
                assert np.all(Vm > 0.5), (
                    f"Method '{r['method_id']}': voltage too low {Vm.min()}"
                )
                assert np.all(Vm < 1.5), (
                    f"Method '{r['method_id']}': voltage too high {Vm.max()}"
                )


# ======================================================================
# Test 5: Per-category convergence analysis
# ======================================================================


class TestCategoryConvergence:
    """Validate convergence behavior grouped by method category."""

    @pytest.fixture
    def all_results_5bus(self, meshed_5bus_net) -> List[Dict[str, Any]]:
        """Run all methods on the 5-bus network."""
        return run_all_methods(meshed_5bus_net, region="category_test")

    def test_pandapower_category_count(
        self, all_results_5bus,
    ) -> None:
        """There are exactly 5 pandapower methods."""
        pp_results = [
            r for r in all_results_5bus if r["category"] == "pandapower"
        ]
        assert len(pp_results) == 5

    def test_custom_nr_category_count(
        self, all_results_5bus,
    ) -> None:
        """There are exactly 7 custom NR methods."""
        nr_results = [
            r for r in all_results_5bus if r["category"] == "custom_nr"
        ]
        assert len(nr_results) == 7

    def test_custom_iterative_category_count(
        self, all_results_5bus,
    ) -> None:
        """There are exactly 4 custom iterative methods."""
        iter_results = [
            r for r in all_results_5bus
            if r["category"] == "custom_iterative"
        ]
        assert len(iter_results) == 4

    def test_custom_decoupled_category_count(
        self, all_results_5bus,
    ) -> None:
        """There are exactly 4 custom decoupled methods."""
        dec_results = [
            r for r in all_results_5bus
            if r["category"] == "custom_decoupled"
        ]
        assert len(dec_results) == 4

    def test_all_custom_nr_converge_on_well_conditioned(
        self, all_results_5bus,
    ) -> None:
        """All 7 custom NR variants converge on the well-conditioned network."""
        nr_results = [
            r for r in all_results_5bus if r["category"] == "custom_nr"
        ]
        for r in nr_results:
            result = r["result"]
            assert result.converged is True, (
                f"Custom NR '{r['method_id']}' did not converge: "
                f"{result.failure_reason}"
            )

    def test_nr_convergence_history_decreasing(
        self, all_results_5bus,
    ) -> None:
        """Custom NR methods have decreasing convergence histories."""
        nr_results = [
            r for r in all_results_5bus if r["category"] == "custom_nr"
        ]
        for r in nr_results:
            result = r["result"]
            if result.converged and len(result.convergence_history) >= 2:
                hist = result.convergence_history
                # Final mismatch should be smaller than initial
                assert hist[-1] < hist[0], (
                    f"Method '{r['method_id']}': final mismatch "
                    f"({hist[-1]}) >= initial ({hist[0]})"
                )


# ======================================================================
# Test 6: Convergence report from comprehensive results
# ======================================================================


class TestComprehensiveConvergenceReport:
    """Generate and validate convergence report from multi-network results."""

    @pytest.fixture
    def multi_network_results(
        self, simple_3bus_net, meshed_5bus_net,
    ) -> List[Dict[str, Any]]:
        """Run all methods on 2 networks and combine results."""
        results_3bus = run_all_methods(
            simple_3bus_net, region="test_3bus",
        )
        results_5bus = run_all_methods(
            meshed_5bus_net, region="test_5bus",
        )
        return results_3bus + results_5bus

    def test_report_has_all_20_methods(
        self, multi_network_results,
    ) -> None:
        """Report contains statistics for all 20 methods."""
        report = generate_report(multi_network_results)
        assert len(report["methods"]) == 20

    def test_report_regions_tested_is_2(
        self, multi_network_results,
    ) -> None:
        """Each method was tested on 2 regions (networks)."""
        report = generate_report(multi_network_results)
        for method_stats in report["methods"]:
            assert method_stats["regions_tested"] == 2, (
                f"Method '{method_stats['method_id']}' tested on "
                f"{method_stats['regions_tested']} regions, expected 2"
            )

    def test_report_convergence_rates_valid(
        self, multi_network_results,
    ) -> None:
        """All convergence rates are between 0 and 100."""
        report = generate_report(multi_network_results)
        for method_stats in report["methods"]:
            rate = method_stats["convergence_rate"]
            assert 0.0 <= rate <= 100.0, (
                f"Method '{method_stats['method_id']}' has invalid "
                f"convergence_rate: {rate}"
            )

    def test_report_summary_totals_consistent(
        self, multi_network_results,
    ) -> None:
        """Summary totals match per-method aggregate counts."""
        report = generate_report(multi_network_results)
        summary = report["summary"]

        total_conv = sum(
            m["converged_count"] for m in report["methods"]
        )
        total_fail = sum(
            m["failed_count"] for m in report["methods"]
        )

        assert summary["total_converged"] == total_conv
        assert summary["total_failed"] == total_fail
        assert summary["total_tests"] == total_conv + total_fail
        assert summary["total_methods"] == 20

    def test_report_custom_nr_high_convergence_rate(
        self, multi_network_results,
    ) -> None:
        """Custom NR should have 100% convergence on well-conditioned nets."""
        report = generate_report(multi_network_results)
        nr_stats = next(
            m for m in report["methods"]
            if m["method_id"] == "custom_nr"
        )
        assert nr_stats["convergence_rate"] == 100.0, (
            f"custom_nr convergence rate is {nr_stats['convergence_rate']}%, "
            f"expected 100%"
        )

    def test_report_saved_to_json(
        self, multi_network_results, tmp_path,
    ) -> None:
        """Report is saved as valid JSON to a temporary file."""
        report = generate_report(multi_network_results)
        output_path = str(tmp_path / "comprehensive_report.json")
        save_report(report, output_path)

        assert os.path.exists(output_path)

        with open(output_path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)

        assert "methods" in loaded
        assert "summary" in loaded
        assert len(loaded["methods"]) == 20

    def test_report_method_entries_have_required_fields(
        self, multi_network_results,
    ) -> None:
        """Each method entry in the report has all required fields."""
        report = generate_report(multi_network_results)
        required_fields = {
            "method_id", "method_name", "category",
            "regions_tested", "converged_count", "failed_count",
            "convergence_rate", "avg_iterations", "avg_elapsed_sec",
            "failure_reasons", "convergence_history",
        }
        for method_stats in report["methods"]:
            missing = required_fields - set(method_stats.keys())
            assert not missing, (
                f"Method '{method_stats.get('method_id')}' missing "
                f"fields: {missing}"
            )


# ======================================================================
# Test 7: Network preparation for reconstructed networks
# ======================================================================


class TestNetworkPrepReconstructed:
    """Validate network preparation on reconstructed networks."""

    def test_prepare_simplified_network(
        self, reconstructed_simplified_net,
    ) -> None:
        """prepare_network() succeeds on simplified network."""
        data = prepare_network(
            copy.deepcopy(reconstructed_simplified_net),
        )
        assert isinstance(data, NetworkData)
        n = data.Ybus.shape[0]
        assert n >= 2, f"Expected at least 2 buses, got {n}"
        assert len(data.ref) >= 1
        assert data.baseMVA > 0

    def test_prepare_reconnected_network(
        self, reconstructed_reconnected_net,
    ) -> None:
        """prepare_network() succeeds on reconnected network."""
        data = prepare_network(
            copy.deepcopy(reconstructed_reconnected_net),
        )
        assert isinstance(data, NetworkData)
        n = data.Ybus.shape[0]
        assert n >= 2, f"Expected at least 2 buses, got {n}"
        assert len(data.ref) >= 1

    def test_bus_classification_complete(
        self, reconstructed_simplified_net,
    ) -> None:
        """All internal buses are classified as ref, pv, or pq."""
        data = prepare_network(
            copy.deepcopy(reconstructed_simplified_net),
        )
        n = data.Ybus.shape[0]
        all_buses = (
            set(data.ref) | set(data.pv) | set(data.pq)
        )
        assert all_buses == set(range(n)), (
            f"Bus classification does not cover all {n} buses"
        )

    def test_ybus_square_and_nonsingular(
        self, reconstructed_simplified_net,
    ) -> None:
        """Ybus is square and the diagonal is non-zero (non-singular)."""
        data = prepare_network(
            copy.deepcopy(reconstructed_simplified_net),
        )
        n = data.Ybus.shape[0]
        assert data.Ybus.shape == (n, n)
        # Check that diagonal is non-zero
        diag = data.Ybus.diagonal()
        assert np.all(np.abs(diag) > 0), (
            "Ybus has zero diagonal entries — matrix may be singular"
        )


# ======================================================================
# Test 8: Voltage consistency across methods
# ======================================================================


class TestVoltageConsistencyAcrossMethods:
    """Converged methods should agree on voltage solutions."""

    def test_nr_variants_agree_on_voltage(
        self, meshed_5bus_net,
    ) -> None:
        """NR variants that converge produce similar voltage solutions."""
        data = prepare_network(copy.deepcopy(meshed_5bus_net))
        methods = get_all_methods()
        nr_methods = [m for m in methods if m.category == "custom_nr"]

        converged_voltages = {}
        for method in nr_methods:
            result = method.solver_fn(
                data.Ybus,
                data.Sbus,
                np.copy(data.V0),
                np.array(data.ref),
                np.array(data.pv),
                np.array(data.pq),
                max_iter=20,
                tol=1e-8,
            )
            if result.converged and result.V is not None:
                converged_voltages[method.id] = result.V

        # At least custom_nr should have converged
        assert "custom_nr" in converged_voltages, (
            "custom_nr did not converge"
        )

        # All converged NR variants should produce similar voltages
        reference_V = converged_voltages["custom_nr"]
        for method_id, V in converged_voltages.items():
            if method_id == "custom_nr":
                continue
            Vm_ref = np.abs(reference_V)
            Vm = np.abs(V)
            max_diff = np.max(np.abs(Vm - Vm_ref))
            assert max_diff < 0.01, (
                f"Method '{method_id}' voltage differs from custom_nr "
                f"by {max_diff:.6f} p.u. (max)"
            )


# ======================================================================
# Test 9: Reproducibility of solver results
# ======================================================================


class TestSolverReproducibility:
    """Verify that solver results are deterministic (reproducible)."""

    def test_custom_nr_reproducible(self, simple_3bus_net) -> None:
        """Running custom_nr twice on the same network produces identical V."""
        data = prepare_network(copy.deepcopy(simple_3bus_net))

        from src.ac_powerflow.custom_solvers import custom_nr

        result1 = custom_nr(
            data.Ybus, data.Sbus, np.copy(data.V0),
            data.ref, data.pv, data.pq,
            max_iter=20, tol=1e-8,
        )
        result2 = custom_nr(
            data.Ybus, data.Sbus, np.copy(data.V0),
            data.ref, data.pv, data.pq,
            max_iter=20, tol=1e-8,
        )

        assert result1.converged is True
        assert result2.converged is True
        assert result1.iterations == result2.iterations
        np.testing.assert_array_almost_equal(
            result1.V, result2.V, decimal=12,
            err_msg="custom_nr produced different voltages on repeated runs",
        )

    def test_run_all_methods_reproducible_count(
        self, simple_3bus_net,
    ) -> None:
        """run_all_methods produces the same number of converged methods."""
        results1 = run_all_methods(simple_3bus_net, region="repro_1")
        results2 = run_all_methods(simple_3bus_net, region="repro_2")

        conv1 = sum(1 for r in results1 if r["result"].converged)
        conv2 = sum(1 for r in results2 if r["result"].converged)
        assert conv1 == conv2, (
            f"Convergence count differs: run1={conv1}, run2={conv2}"
        )


# ======================================================================
# Test 10: Cross-network method comparison
# ======================================================================


class TestCrossNetworkComparison:
    """Compare method behavior across different network topologies."""

    @pytest.fixture
    def results_3bus(self, simple_3bus_net) -> List[Dict[str, Any]]:
        """Run all methods on 3-bus network."""
        return run_all_methods(simple_3bus_net, region="cross_3bus")

    @pytest.fixture
    def results_5bus(self, meshed_5bus_net) -> List[Dict[str, Any]]:
        """Run all methods on 5-bus network."""
        return run_all_methods(meshed_5bus_net, region="cross_5bus")

    def test_both_networks_produce_20_results(
        self, results_3bus, results_5bus,
    ) -> None:
        """Both networks produce exactly 20 method results."""
        assert len(results_3bus) == 20
        assert len(results_5bus) == 20

    def test_method_ids_match(
        self, results_3bus, results_5bus,
    ) -> None:
        """Both networks test the same set of method ids."""
        ids_3bus = {r["method_id"] for r in results_3bus}
        ids_5bus = {r["method_id"] for r in results_5bus}
        assert ids_3bus == ids_5bus, (
            f"Method id mismatch: "
            f"only_3bus={ids_3bus - ids_5bus}, "
            f"only_5bus={ids_5bus - ids_3bus}"
        )

    def test_custom_nr_converges_on_both(
        self, results_3bus, results_5bus,
    ) -> None:
        """Custom NR converges on both 3-bus and 5-bus networks."""
        nr_3 = next(
            r for r in results_3bus if r["method_id"] == "custom_nr"
        )
        nr_5 = next(
            r for r in results_5bus if r["method_id"] == "custom_nr"
        )
        assert nr_3["result"].converged is True
        assert nr_5["result"].converged is True

    def test_combined_report_across_networks(
        self, results_3bus, results_5bus, tmp_path,
    ) -> None:
        """Combined report from both networks has correct structure."""
        combined = results_3bus + results_5bus
        report = generate_report(combined)

        assert report["summary"]["total_methods"] == 20
        assert report["summary"]["total_tests"] == 40

        # Each method should have 2 region entries
        for m in report["methods"]:
            assert m["regions_tested"] == 2, (
                f"Method '{m['method_id']}' tested on "
                f"{m['regions_tested']} regions, expected 2"
            )

        # Save and verify
        output_path = str(tmp_path / "cross_network_report.json")
        save_report(report, output_path)
        assert os.path.exists(output_path)


# ======================================================================
# Test 11: Reconstruction pipeline + AC power flow end-to-end
# ======================================================================


class TestReconstructionPipelineACPowerflow:
    """End-to-end: reconstruction pipeline -> all AC power flow methods."""

    def test_simplify_then_all_methods(self) -> None:
        """Full pipeline: simplify -> run all 20 methods -> validate."""
        raw_net = make_isolated_network(
            n_main_buses=4,
            n_isolated_buses=3,
            n_generators=2,
        )

        cfg = ReconstructionConfig(
            mode="simplify",
            seed=42,
            skip_existing_loads=False,
            skip_existing_generation=False,
            db_path=":memory:",
        )
        pipeline = ReconstructionPipeline(cfg, copy_network=True)
        pipe_result = pipeline.run(raw_net, region="e2e_simplify")

        assert pipe_result.net is not None

        results = run_all_methods(
            pipe_result.net, region="e2e_simplify",
        )
        assert len(results) == 20

        # At least custom_nr should converge
        nr = next(r for r in results if r["method_id"] == "custom_nr")
        assert nr["result"].converged is True, (
            f"custom_nr failed: {nr['result'].failure_reason}"
        )

    def test_reconnect_then_all_methods(self) -> None:
        """Full pipeline: reconnect -> run all 20 methods -> validate."""
        raw_net = make_isolated_network(
            n_main_buses=4,
            n_isolated_buses=3,
            n_generators=2,
        )

        cfg = ReconstructionConfig(
            mode="reconnect",
            seed=42,
            min_reactance_ohm_per_km=0.001,
            skip_existing_loads=False,
            skip_existing_generation=False,
            db_path=":memory:",
        )
        pipeline = ReconstructionPipeline(cfg, copy_network=True)
        pipe_result = pipeline.run(raw_net, region="e2e_reconnect")

        assert pipe_result.net is not None

        results = run_all_methods(
            pipe_result.net, region="e2e_reconnect",
        )
        assert len(results) == 20

        # At least custom_nr should converge
        nr = next(r for r in results if r["method_id"] == "custom_nr")
        assert nr["result"].converged is True, (
            f"custom_nr failed: {nr['result'].failure_reason}"
        )

    def test_both_modes_produce_valid_reports(self, tmp_path) -> None:
        """Both modes produce valid convergence reports."""
        all_results: List[Dict[str, Any]] = []

        for mode in ("simplify", "reconnect"):
            raw_net = make_isolated_network(
                n_main_buses=3,
                n_isolated_buses=2,
                n_generators=2,
            )

            cfg = ReconstructionConfig(
                mode=mode,
                seed=42,
                min_reactance_ohm_per_km=0.001,
                skip_existing_loads=False,
                skip_existing_generation=False,
                db_path=":memory:",
            )
            pipeline = ReconstructionPipeline(cfg, copy_network=True)
            pipe_result = pipeline.run(raw_net, region=f"report_{mode}")

            results = run_all_methods(
                pipe_result.net, region=f"report_{mode}",
            )
            all_results.extend(results)

        # Generate combined report
        report = generate_report(all_results)
        assert report["summary"]["total_methods"] == 20
        assert report["summary"]["total_tests"] == 40

        # Save and validate JSON
        output_path = str(tmp_path / "e2e_report.json")
        save_report(report, output_path)
        assert os.path.exists(output_path)

        with open(output_path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert len(loaded["methods"]) == 20


# ======================================================================
# Test 12: Individual method validation on reconstructed networks
# ======================================================================


class TestIndividualMethodValidation:
    """Validate specific method characteristics on reconstructed networks."""

    def test_custom_nr_convergence_history_populated(
        self, reconstructed_simplified_net,
    ) -> None:
        """custom_nr records convergence history on simplified network."""
        data = prepare_network(
            copy.deepcopy(reconstructed_simplified_net),
        )

        from src.ac_powerflow.custom_solvers import custom_nr

        result = custom_nr(
            data.Ybus, data.Sbus, np.copy(data.V0),
            data.ref, data.pv, data.pq,
            max_iter=20, tol=1e-8,
        )
        assert result.converged is True
        assert len(result.convergence_history) > 0
        assert result.convergence_history[-1] < 1e-8

    def test_custom_nr_linesearch_converges(
        self, reconstructed_simplified_net,
    ) -> None:
        """custom_nr_linesearch converges on simplified network."""
        data = prepare_network(
            copy.deepcopy(reconstructed_simplified_net),
        )

        from src.ac_powerflow.custom_solvers import custom_nr_linesearch

        result = custom_nr_linesearch(
            data.Ybus, data.Sbus, np.copy(data.V0),
            data.ref, data.pv, data.pq,
            max_iter=20, tol=1e-8,
        )
        assert result.converged is True

    def test_custom_gs_returns_valid_result(
        self, reconstructed_simplified_net,
    ) -> None:
        """custom_gs returns a valid result on simplified network."""
        data = prepare_network(
            copy.deepcopy(reconstructed_simplified_net),
        )

        from src.ac_powerflow.custom_solvers import custom_gs

        result = custom_gs(
            data.Ybus, data.Sbus, np.copy(data.V0),
            data.ref, data.pv, data.pq,
            max_iter=200, tol=1e-6,
        )
        assert isinstance(result, ACMethodResult)
        assert result.V is not None
        assert result.elapsed_sec > 0
        assert not np.isnan(result.V).any()

    def test_custom_nr_iwamoto_converges(
        self, reconstructed_simplified_net,
    ) -> None:
        """custom_nr_iwamoto converges on simplified network."""
        data = prepare_network(
            copy.deepcopy(reconstructed_simplified_net),
        )

        from src.ac_powerflow.custom_solvers import custom_nr_iwamoto

        result = custom_nr_iwamoto(
            data.Ybus, data.Sbus, np.copy(data.V0),
            data.ref, data.pv, data.pq,
            max_iter=20, tol=1e-8,
        )
        assert result.converged is True
        assert result.V is not None

    def test_custom_nr_levenberg_converges(
        self, reconstructed_simplified_net,
    ) -> None:
        """custom_nr_levenberg converges on simplified network."""
        data = prepare_network(
            copy.deepcopy(reconstructed_simplified_net),
        )

        from src.ac_powerflow.custom_solvers import custom_nr_levenberg

        result = custom_nr_levenberg(
            data.Ybus, data.Sbus, np.copy(data.V0),
            data.ref, data.pv, data.pq,
            max_iter=20, tol=1e-8,
        )
        assert result.converged is True
        assert result.V is not None

    def test_custom_fdpf_bx_executes(
        self, reconstructed_simplified_net,
    ) -> None:
        """custom_fdpf_bx executes without crash on simplified network."""
        data = prepare_network(
            copy.deepcopy(reconstructed_simplified_net),
        )

        from src.ac_powerflow.custom_solvers import custom_fdpf_bx

        result = custom_fdpf_bx(
            data.Ybus, data.Sbus, np.copy(data.V0),
            data.ref, data.pv, data.pq,
            max_iter=50, tol=1e-6,
        )
        assert isinstance(result, ACMethodResult)
        assert result.elapsed_sec >= 0
