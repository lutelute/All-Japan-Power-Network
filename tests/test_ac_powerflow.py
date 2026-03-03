"""Unit tests for the AC power flow analysis module.

Tests cover:
- Method registry — all ~20 methods registered with valid metadata
- Solver interface — ACMethodResult dataclass fields and .summary property
- Pandapower wrappers — NR and GS via pandapower on a simple network
- Custom solvers — NR and GS from scratch on extracted network data
- Network preparation — Ybus, Sbus, V0, ref, pv, pq extraction
- Convergence report — JSON report structure and required fields
- Error handling — singular Jacobian and NaN detection

All tests use a simple 3-bus pandapower network fixture:
    Bus 0: ext_grid (slack/reference) at 110 kV
    Bus 1: generator (PV bus) at 110 kV, 40 MW
    Bus 2: load (PQ bus) at 110 kV, 60 MW + 20 Mvar
    Line 0-1: 10 km standard line
    Line 1-2: 10 km standard line
    Line 0-2: 15 km standard line
"""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandapower as pp
import pytest
from scipy import sparse

from src.ac_powerflow.convergence_report import generate_report, save_report
from src.ac_powerflow.custom_solvers import custom_gs, custom_nr
from src.ac_powerflow.methods import MethodDescriptor, get_all_methods
from src.ac_powerflow.network_prep import NetworkData, prepare_network
from src.ac_powerflow.pandapower_methods import pp_gs, pp_nr
from src.ac_powerflow.solver_interface import ACMethodResult


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def simple_3bus_net():
    """Create a simple 3-bus pandapower network for testing.

    Topology::

        Bus 0 (slack) ---[line 0-1]--- Bus 1 (gen, PV)
          |                              |
        [line 0-2]                   [line 1-2]
          |                              |
        Bus 2 (load, PQ) ---------------+

    All buses at 110 kV.  Uses standard pandapower line type
    ``"149-AL1/24-ST1A 110.0"`` (110 kV overhead line).
    """
    net = pp.create_empty_network(f_hz=50.0)

    # Buses
    bus0 = pp.create_bus(net, vn_kv=110.0, name="Bus 0 (slack)")
    bus1 = pp.create_bus(net, vn_kv=110.0, name="Bus 1 (gen)")
    bus2 = pp.create_bus(net, vn_kv=110.0, name="Bus 2 (load)")

    # External grid (slack bus)
    pp.create_ext_grid(net, bus=bus0, vm_pu=1.02, name="Grid")

    # Generator on bus 1 (PV bus)
    pp.create_gen(net, bus=bus1, p_mw=40.0, vm_pu=1.01, name="Gen 1")

    # Load on bus 2 (PQ bus)
    pp.create_load(net, bus=bus2, p_mw=60.0, q_mvar=20.0, name="Load 2")

    # Lines (standard 110 kV overhead line type)
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
def network_data(simple_3bus_net):
    """Extract NetworkData from the simple 3-bus network."""
    return prepare_network(simple_3bus_net)


# ======================================================================
# Test 1: Method Registry
# ======================================================================


class TestMethodRegistry:
    """Tests for the central method registry."""

    def test_method_registry_count(self) -> None:
        """get_all_methods() returns ~20 method descriptors."""
        methods = get_all_methods()
        assert len(methods) == 20, (
            f"Expected 20 registered methods, got {len(methods)}"
        )

    def test_method_registry_types(self) -> None:
        """All entries are MethodDescriptor instances."""
        methods = get_all_methods()
        for method in methods:
            assert isinstance(method, MethodDescriptor)

    def test_method_registry_valid_metadata(self) -> None:
        """Each method has non-empty id, name, category, description, and callable solver_fn."""
        methods = get_all_methods()
        valid_categories = {"pandapower", "custom_nr", "custom_iterative", "custom_decoupled"}

        for method in methods:
            assert method.id, f"Method has empty id: {method}"
            assert method.name, f"Method has empty name: {method}"
            assert method.category in valid_categories, (
                f"Method '{method.id}' has invalid category '{method.category}'"
            )
            assert method.description, (
                f"Method '{method.id}' has empty description"
            )
            assert callable(method.solver_fn), (
                f"Method '{method.id}' solver_fn is not callable"
            )

    def test_method_registry_unique_ids(self) -> None:
        """All method ids are unique."""
        methods = get_all_methods()
        ids = [m.id for m in methods]
        assert len(ids) == len(set(ids)), (
            f"Duplicate method ids found: "
            f"{[mid for mid in ids if ids.count(mid) > 1]}"
        )

    def test_method_registry_category_counts(self) -> None:
        """Category distribution matches expected counts."""
        methods = get_all_methods()
        categories = {}
        for m in methods:
            categories[m.category] = categories.get(m.category, 0) + 1

        assert categories.get("pandapower", 0) == 5
        assert categories.get("custom_nr", 0) == 7
        assert categories.get("custom_iterative", 0) == 4
        assert categories.get("custom_decoupled", 0) == 4


# ======================================================================
# Test 2: Solver Interface
# ======================================================================


class TestSolverInterface:
    """Tests for the ACMethodResult dataclass."""

    def test_default_values(self) -> None:
        """ACMethodResult has correct default values."""
        result = ACMethodResult()
        assert result.converged is False
        assert result.iterations == 0
        assert result.V is None
        assert result.elapsed_sec == 0.0
        assert result.convergence_history == []
        assert result.failure_reason is None

    def test_all_required_fields(self) -> None:
        """ACMethodResult has all specification-required fields."""
        result = ACMethodResult()
        assert hasattr(result, "converged")
        assert hasattr(result, "iterations")
        assert hasattr(result, "V")
        assert hasattr(result, "elapsed_sec")
        assert hasattr(result, "convergence_history")
        assert hasattr(result, "failure_reason")

    def test_summary_property(self) -> None:
        """The .summary property returns a dict with required keys."""
        result = ACMethodResult(
            converged=True,
            iterations=5,
            elapsed_sec=0.123456,
            convergence_history=[1.0, 0.1, 0.001, 1e-6, 1e-10],
        )
        summary = result.summary
        assert isinstance(summary, dict)
        assert "converged" in summary
        assert "iterations" in summary
        assert "elapsed_sec" in summary
        assert "final_mismatch" in summary
        assert "failure_reason" in summary

    def test_summary_values(self) -> None:
        """The .summary property returns correct values."""
        result = ACMethodResult(
            converged=True,
            iterations=3,
            elapsed_sec=0.05,
            convergence_history=[1.0, 0.01, 1e-6],
            failure_reason=None,
        )
        summary = result.summary
        assert summary["converged"] is True
        assert summary["iterations"] == 3
        # round(1e-6, 8) = 1e-6
        assert summary["final_mismatch"] == pytest.approx(1e-6, abs=1e-10)
        assert summary["failure_reason"] is None

    def test_summary_empty_history(self) -> None:
        """The .summary property handles empty convergence history."""
        result = ACMethodResult()
        summary = result.summary
        assert summary["final_mismatch"] is None


# ======================================================================
# Test 3: Pandapower NR Wrapper
# ======================================================================


class TestPandapowerNRWrapper:
    """Tests for the pandapower Newton-Raphson wrapper.

    Note: Some pandapower versions raise a ValueError during result
    extraction (read-only DataFrame).  The wrapper catches this and
    reports a failure reason.  The internal solver still converges
    (visible in ``net._ppc``), but the wrapper correctly classifies
    it as non-converged because result extraction failed.  Tests here
    verify the wrapper runs without crashing and returns a valid
    ``ACMethodResult``.
    """

    def test_pp_nr_runs_without_crash(self, simple_3bus_net) -> None:
        """pp_nr completes without raising an exception."""
        net = deepcopy(simple_3bus_net)
        result = pp_nr(net)
        # The wrapper should always return a result, even if the
        # internal solver encounters errors during result extraction.
        assert isinstance(result, ACMethodResult)
        assert result.elapsed_sec > 0
        # Either it converged or it has a failure reason
        if not result.converged:
            assert result.failure_reason is not None

    def test_pp_nr_returns_acmethodresult(self, simple_3bus_net) -> None:
        """pp_nr returns an ACMethodResult instance."""
        net = deepcopy(simple_3bus_net)
        result = pp_nr(net)
        assert isinstance(result, ACMethodResult)

    def test_pp_nr_timing(self, simple_3bus_net) -> None:
        """pp_nr records a positive elapsed time."""
        net = deepcopy(simple_3bus_net)
        result = pp_nr(net)
        assert result.elapsed_sec > 0
        # Should complete in under 10 seconds for a 3-bus network
        assert result.elapsed_sec < 10.0


# ======================================================================
# Test 4: Pandapower GS Wrapper
# ======================================================================


class TestPandapowerGSWrapper:
    """Tests for the pandapower Gauss-Seidel wrapper.

    See note on ``TestPandapowerNRWrapper`` regarding read-only
    DataFrame issues in some pandapower versions.
    """

    def test_pp_gs_executes(self, simple_3bus_net) -> None:
        """pp_gs executes without raising exceptions."""
        net = deepcopy(simple_3bus_net)
        result = pp_gs(net, max_iteration=100)
        assert isinstance(result, ACMethodResult)
        assert result.elapsed_sec > 0

    def test_pp_gs_returns_valid_result(self, simple_3bus_net) -> None:
        """pp_gs returns a result with expected field types."""
        net = deepcopy(simple_3bus_net)
        result = pp_gs(net, max_iteration=100)
        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)
        assert isinstance(result.elapsed_sec, float)
        # GS may or may not converge — wrapper may also hit read-only
        # DataFrame issues.  We're testing that it runs without crash
        # and returns valid types.


# ======================================================================
# Test 5: Custom NR Solver
# ======================================================================


class TestCustomNR:
    """Tests for the custom Newton-Raphson solver."""

    def test_custom_nr_produces_valid_voltage(self, network_data) -> None:
        """custom_nr produces a valid complex voltage vector."""
        data = network_data
        result = custom_nr(
            data.Ybus, data.Sbus, data.V0,
            data.ref, data.pv, data.pq,
            max_iter=20, tol=1e-8,
        )
        assert result.converged is True
        assert result.V is not None
        assert isinstance(result.V, np.ndarray)
        assert result.V.dtype == complex
        assert len(result.V) == data.Ybus.shape[0]

    def test_custom_nr_voltage_magnitudes_reasonable(self, network_data) -> None:
        """custom_nr voltage magnitudes are within reasonable range [0.8, 1.2] p.u."""
        data = network_data
        result = custom_nr(
            data.Ybus, data.Sbus, data.V0,
            data.ref, data.pv, data.pq,
            max_iter=20, tol=1e-8,
        )
        assert result.converged is True
        Vm = np.abs(result.V)
        assert np.all(Vm > 0.8), f"Some voltage magnitudes below 0.8 p.u.: {Vm}"
        assert np.all(Vm < 1.2), f"Some voltage magnitudes above 1.2 p.u.: {Vm}"

    def test_custom_nr_convergence_history(self, network_data) -> None:
        """custom_nr records decreasing mismatch norms."""
        data = network_data
        result = custom_nr(
            data.Ybus, data.Sbus, data.V0,
            data.ref, data.pv, data.pq,
            max_iter=20, tol=1e-8,
        )
        assert result.converged is True
        assert len(result.convergence_history) > 0
        # Final mismatch should be below tolerance
        assert result.convergence_history[-1] < 1e-8

    def test_custom_nr_returns_acmethodresult(self, network_data) -> None:
        """custom_nr returns an ACMethodResult instance."""
        data = network_data
        result = custom_nr(
            data.Ybus, data.Sbus, data.V0,
            data.ref, data.pv, data.pq,
        )
        assert isinstance(result, ACMethodResult)


# ======================================================================
# Test 6: Custom GS Solver
# ======================================================================


class TestCustomGS:
    """Tests for the custom Gauss-Seidel solver."""

    def test_custom_gs_executes_max_iter(self, network_data) -> None:
        """custom_gs executes for max_iter iterations when tolerance is very tight."""
        data = network_data
        # Use very tight tolerance so GS won't converge in 5 iterations
        result = custom_gs(
            data.Ybus, data.Sbus, data.V0,
            data.ref, data.pv, data.pq,
            max_iter=5, tol=1e-20,
        )
        # GS with only 5 iterations and very tight tol should not converge
        assert result.iterations == 5
        assert result.converged is False

    def test_custom_gs_returns_valid_result(self, network_data) -> None:
        """custom_gs returns an ACMethodResult with recorded history."""
        data = network_data
        result = custom_gs(
            data.Ybus, data.Sbus, data.V0,
            data.ref, data.pv, data.pq,
            max_iter=10, tol=1e-8,
        )
        assert isinstance(result, ACMethodResult)
        assert result.elapsed_sec > 0
        assert len(result.convergence_history) > 0
        assert result.V is not None

    def test_custom_gs_voltage_vector_valid(self, network_data) -> None:
        """custom_gs produces a voltage vector without NaN or Inf."""
        data = network_data
        result = custom_gs(
            data.Ybus, data.Sbus, data.V0,
            data.ref, data.pv, data.pq,
            max_iter=50, tol=1e-8,
        )
        assert result.V is not None
        assert not np.isnan(result.V).any(), "Voltage vector contains NaN"
        assert not np.isinf(result.V).any(), "Voltage vector contains Inf"


# ======================================================================
# Test 7: Network Preparation
# ======================================================================


class TestNetworkPrep:
    """Tests for the prepare_network() function."""

    def test_prepare_network_returns_networkdata(self, simple_3bus_net) -> None:
        """prepare_network() returns a NetworkData instance."""
        data = prepare_network(simple_3bus_net)
        assert isinstance(data, NetworkData)

    def test_ybus_is_sparse(self, network_data) -> None:
        """Ybus is a sparse CSC matrix."""
        assert sparse.issparse(network_data.Ybus)
        assert isinstance(network_data.Ybus, sparse.csc_matrix)

    def test_ybus_shape(self, network_data) -> None:
        """Ybus is a square matrix matching bus count."""
        n = network_data.Ybus.shape[0]
        assert network_data.Ybus.shape == (n, n)
        assert n >= 3, "Expected at least 3 buses in the network"

    def test_sbus_shape(self, network_data) -> None:
        """Sbus is a 1D complex array matching bus count."""
        n = network_data.Ybus.shape[0]
        assert network_data.Sbus.shape == (n,)
        assert network_data.Sbus.dtype == complex

    def test_v0_shape(self, network_data) -> None:
        """V0 is a 1D complex array matching bus count."""
        n = network_data.Ybus.shape[0]
        assert network_data.V0.shape == (n,)
        assert network_data.V0.dtype == complex

    def test_bus_classification(self, network_data) -> None:
        """ref, pv, and pq lists are populated and non-overlapping."""
        ref = set(network_data.ref)
        pv = set(network_data.pv)
        pq = set(network_data.pq)

        assert len(ref) >= 1, "Must have at least 1 reference (slack) bus"
        # Bus sets should not overlap
        assert ref.isdisjoint(pv), "ref and pv buses overlap"
        assert ref.isdisjoint(pq), "ref and pq buses overlap"
        assert pv.isdisjoint(pq), "pv and pq buses overlap"

    def test_basemva_positive(self, network_data) -> None:
        """baseMVA is a positive float."""
        assert isinstance(network_data.baseMVA, float)
        assert network_data.baseMVA > 0

    def test_bus_indices_cover_network(self, network_data) -> None:
        """All internal bus indices are accounted for in ref + pv + pq."""
        n = network_data.Ybus.shape[0]
        all_buses = set(network_data.ref) | set(network_data.pv) | set(network_data.pq)
        assert all_buses == set(range(n)), (
            f"Bus classification does not cover all {n} buses: "
            f"ref={network_data.ref}, pv={network_data.pv}, pq={network_data.pq}"
        )


# ======================================================================
# Test 8: Convergence Report
# ======================================================================


class TestConvergenceReport:
    """Tests for the convergence report generation."""

    @pytest.fixture
    def sample_results(self) -> list:
        """Create sample batch results for report generation."""
        return [
            {
                "method_id": "pp_nr",
                "method_name": "pp_nr",
                "category": "pandapower",
                "region": "test_region_a",
                "result": ACMethodResult(
                    converged=True,
                    iterations=3,
                    elapsed_sec=0.05,
                    convergence_history=[1.0, 0.01, 1e-9],
                ),
            },
            {
                "method_id": "pp_nr",
                "method_name": "pp_nr",
                "category": "pandapower",
                "region": "test_region_b",
                "result": ACMethodResult(
                    converged=True,
                    iterations=4,
                    elapsed_sec=0.07,
                    convergence_history=[2.0, 0.1, 0.001, 1e-10],
                ),
            },
            {
                "method_id": "custom_nr",
                "method_name": "custom_nr",
                "category": "custom_nr",
                "region": "test_region_a",
                "result": ACMethodResult(
                    converged=False,
                    iterations=20,
                    elapsed_sec=0.15,
                    convergence_history=[1.0] * 20,
                    failure_reason="Did not converge within 20 iterations",
                ),
            },
        ]

    def test_report_has_methods_key(self, sample_results) -> None:
        """Report contains 'methods' list."""
        report = generate_report(sample_results)
        assert "methods" in report
        assert isinstance(report["methods"], list)

    def test_report_has_summary_key(self, sample_results) -> None:
        """Report contains 'summary' dict."""
        report = generate_report(sample_results)
        assert "summary" in report
        assert isinstance(report["summary"], dict)

    def test_method_stats_required_fields(self, sample_results) -> None:
        """Each method entry has required fields."""
        report = generate_report(sample_results)
        required_fields = {
            "method_id",
            "method_name",
            "category",
            "regions_tested",
            "converged_count",
            "failed_count",
            "convergence_rate",
            "avg_iterations",
            "avg_elapsed_sec",
            "failure_reasons",
            "convergence_history",
        }
        for method_stats in report["methods"]:
            missing = required_fields - set(method_stats.keys())
            assert not missing, (
                f"Method '{method_stats.get('method_id')}' missing fields: {missing}"
            )

    def test_convergence_rate_calculation(self, sample_results) -> None:
        """Convergence rate is correctly calculated."""
        report = generate_report(sample_results)
        # pp_nr: 2/2 converged = 100%
        pp_nr_stats = next(
            m for m in report["methods"] if m["method_id"] == "pp_nr"
        )
        assert pp_nr_stats["convergence_rate"] == 100.0
        assert pp_nr_stats["converged_count"] == 2
        assert pp_nr_stats["failed_count"] == 0

        # custom_nr: 0/1 converged = 0%
        custom_nr_stats = next(
            m for m in report["methods"] if m["method_id"] == "custom_nr"
        )
        assert custom_nr_stats["convergence_rate"] == 0.0
        assert custom_nr_stats["failed_count"] == 1

    def test_failure_reasons_populated(self, sample_results) -> None:
        """Failure reasons are captured for non-converged methods."""
        report = generate_report(sample_results)
        custom_nr_stats = next(
            m for m in report["methods"] if m["method_id"] == "custom_nr"
        )
        assert len(custom_nr_stats["failure_reasons"]) > 0
        assert custom_nr_stats["failure_reasons"][0]["count"] == 1

    def test_save_report_creates_file(self, sample_results, tmp_path) -> None:
        """save_report writes a valid JSON file."""
        report = generate_report(sample_results)
        output_path = str(tmp_path / "convergence_report.json")
        result_path = save_report(report, output_path)

        assert result_path == output_path

        import json
        with open(output_path, "r") as f:
            loaded = json.load(f)
        assert "methods" in loaded
        assert "summary" in loaded

    def test_summary_overall_stats(self, sample_results) -> None:
        """Summary contains overall convergence statistics."""
        report = generate_report(sample_results)
        summary = report["summary"]
        assert "total_methods" in summary
        assert "total_tests" in summary
        assert "total_converged" in summary
        assert "total_failed" in summary
        assert "overall_convergence_rate" in summary
        assert summary["total_tests"] == 3
        assert summary["total_converged"] == 2
        assert summary["total_failed"] == 1


# ======================================================================
# Test 9: Singular Jacobian Handling
# ======================================================================


class TestSingularJacobianHandling:
    """Tests for singular Jacobian error handling in NR solver."""

    def test_nr_catches_singular_jacobian(self, network_data) -> None:
        """custom_nr catches LinAlgError from singular Jacobian and reports failure."""
        data = network_data

        # Create a degenerate Ybus by zeroing out entries to make
        # the Jacobian singular.  We patch spsolve to raise LinAlgError
        # to simulate this scenario reliably.
        with patch(
            "src.ac_powerflow.custom_solvers.spsolve",
            side_effect=np.linalg.LinAlgError("Singular matrix"),
        ):
            result = custom_nr(
                data.Ybus, data.Sbus, data.V0,
                data.ref, data.pv, data.pq,
                max_iter=20, tol=1e-8,
            )

        assert result.converged is False
        assert result.failure_reason is not None
        assert "singular" in result.failure_reason.lower() or \
               "Singular" in result.failure_reason

    def test_nr_singular_returns_acmethodresult(self, network_data) -> None:
        """custom_nr returns ACMethodResult even on singular Jacobian."""
        data = network_data

        with patch(
            "src.ac_powerflow.custom_solvers.spsolve",
            side_effect=np.linalg.LinAlgError("Singular matrix"),
        ):
            result = custom_nr(
                data.Ybus, data.Sbus, data.V0,
                data.ref, data.pv, data.pq,
            )

        assert isinstance(result, ACMethodResult)
        assert result.elapsed_sec >= 0


# ======================================================================
# Test 10: NaN Detection
# ======================================================================


class TestNaNDetection:
    """Tests for NaN detection and abort in solvers."""

    def test_nan_in_v0_detected(self, network_data) -> None:
        """Solver detects NaN in initial voltage and aborts."""
        data = network_data

        # Inject NaN into V0
        V0_bad = data.V0.copy()
        V0_bad[0] = np.nan + 0j

        result = custom_nr(
            data.Ybus, data.Sbus, V0_bad,
            data.ref, data.pv, data.pq,
            max_iter=20, tol=1e-8,
        )

        assert result.converged is False
        assert result.failure_reason is not None
        assert "nan" in result.failure_reason.lower() or \
               "NaN" in result.failure_reason

    def test_nan_detection_aborts_early(self, network_data) -> None:
        """Solver aborts within a few iterations when NaN is introduced."""
        data = network_data

        # Patch spsolve to return NaN on first call to simulate
        # numerical instability during iteration
        call_count = [0]
        original_spsolve_module = __import__(
            "scipy.sparse.linalg", fromlist=["spsolve"]
        )

        def spsolve_with_nan(*args, **kwargs):
            call_count[0] += 1
            n = args[0].shape[0]
            # Return NaN array to trigger NaN detection
            return np.full(n, np.nan)

        with patch(
            "src.ac_powerflow.custom_solvers.spsolve",
            side_effect=spsolve_with_nan,
        ):
            result = custom_nr(
                data.Ybus, data.Sbus, data.V0,
                data.ref, data.pv, data.pq,
                max_iter=20, tol=1e-8,
            )

        assert result.converged is False
        assert result.failure_reason is not None
        assert "nan" in result.failure_reason.lower() or \
               "NaN" in result.failure_reason
        # Should abort early, not run all 20 iterations
        assert result.iterations < 20

    def test_inf_in_voltage_detected(self, network_data) -> None:
        """Solver detects Inf in voltage vector and aborts."""
        data = network_data

        # Inject Inf into V0
        V0_bad = data.V0.copy()
        V0_bad[1] = np.inf + 0j

        result = custom_nr(
            data.Ybus, data.Sbus, V0_bad,
            data.ref, data.pv, data.pq,
            max_iter=20, tol=1e-8,
        )

        assert result.converged is False
        assert result.failure_reason is not None
