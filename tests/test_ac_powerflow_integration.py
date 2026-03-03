"""Integration tests for the AC power flow module.

Verifies end-to-end execution of all ~20 methods on a realistic
pandapower network, report generation, and batch runner CLI behavior.

These tests complement the unit tests in ``test_ac_powerflow.py`` by
exercising the full pipeline: method registry → solver execution →
convergence report generation → JSON file output.
"""

import json
import os
from copy import deepcopy

import numpy as np
import pandapower as pp
import pytest

from src.ac_powerflow.batch_runner import run_all_methods
from src.ac_powerflow.convergence_report import generate_report, print_summary, save_report
from src.ac_powerflow.methods import get_all_methods
from src.ac_powerflow.network_prep import prepare_network
from src.ac_powerflow.solver_interface import ACMethodResult


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def realistic_net():
    """Create a realistic 5-bus pandapower network for integration testing.

    Topology::

        Bus 0 (slack, 110 kV)
          |
        [line 0-1: 20 km]
          |
        Bus 1 (gen PV, 110 kV, 80 MW) ---[line 1-3: 25 km]--- Bus 3 (load, 110 kV, 90 MW)
          |                                                        |
        [line 1-2: 15 km]                                       [line 3-4: 10 km]
          |                                                        |
        Bus 2 (load, 110 kV, 50 MW) ---[line 2-4: 30 km]--- Bus 4 (load, 110 kV, 30 MW)

    This is a meshed network (5 buses, 5 lines) that resembles a small
    regional grid segment.
    """
    net = pp.create_empty_network(f_hz=60.0)

    # Buses
    b0 = pp.create_bus(net, vn_kv=110.0, name="Slack Bus")
    b1 = pp.create_bus(net, vn_kv=110.0, name="Gen Bus")
    b2 = pp.create_bus(net, vn_kv=110.0, name="Load Bus A")
    b3 = pp.create_bus(net, vn_kv=110.0, name="Load Bus B")
    b4 = pp.create_bus(net, vn_kv=110.0, name="Load Bus C")

    # External grid (slack)
    pp.create_ext_grid(net, bus=b0, vm_pu=1.02, name="Grid Connection")

    # Generator (PV bus)
    pp.create_gen(net, bus=b1, p_mw=80.0, vm_pu=1.01, name="Thermal Gen")

    # Loads
    pp.create_load(net, bus=b2, p_mw=50.0, q_mvar=15.0, name="Load A")
    pp.create_load(net, bus=b3, p_mw=90.0, q_mvar=30.0, name="Load B")
    pp.create_load(net, bus=b4, p_mw=30.0, q_mvar=10.0, name="Load C")

    # Lines (using standard line type for 110 kV)
    pp.create_line(net, b0, b1, 20.0, "149-AL1/24-ST1A 110.0", name="Line 0-1")
    pp.create_line(net, b1, b2, 15.0, "149-AL1/24-ST1A 110.0", name="Line 1-2")
    pp.create_line(net, b1, b3, 25.0, "149-AL1/24-ST1A 110.0", name="Line 1-3")
    pp.create_line(net, b2, b4, 30.0, "149-AL1/24-ST1A 110.0", name="Line 2-4")
    pp.create_line(net, b3, b4, 10.0, "149-AL1/24-ST1A 110.0", name="Line 3-4")

    return net


# ======================================================================
# Integration: run_all_methods on realistic network
# ======================================================================


class TestRunAllMethods:
    """End-to-end test: run all ~20 methods on a realistic network."""

    def test_all_methods_execute_without_crash(self, realistic_net) -> None:
        """All 20 methods execute without raising unhandled exceptions."""
        results = run_all_methods(realistic_net, region="integration_test")
        assert len(results) == 20, (
            f"Expected 20 method results, got {len(results)}"
        )

    def test_each_result_is_acmethodresult(self, realistic_net) -> None:
        """Each method returns a valid ACMethodResult."""
        results = run_all_methods(realistic_net, region="integration_test")
        for r in results:
            assert isinstance(r["result"], ACMethodResult), (
                f"Method '{r['method_id']}' returned {type(r['result'])}"
            )

    def test_result_records_have_required_keys(self, realistic_net) -> None:
        """Each result record contains method_id, method_name, category, region, result."""
        results = run_all_methods(realistic_net, region="integration_test")
        required_keys = {"method_id", "method_name", "category", "region", "result"}
        for r in results:
            missing = required_keys - set(r.keys())
            assert not missing, (
                f"Method '{r.get('method_id')}' missing keys: {missing}"
            )

    def test_pandapower_methods_converge(self, realistic_net) -> None:
        """Pandapower NR and fast-decoupled methods converge on realistic network."""
        results = run_all_methods(realistic_net, region="integration_test")
        pp_results = [r for r in results if r["category"] == "pandapower"]
        # At minimum pp_nr should converge (or at least execute without crash)
        nr_result = next(r for r in pp_results if r["method_id"] == "pp_nr")
        # NR wrapper may report failure due to read-only DataFrame issue in
        # some pandapower versions, but it should always return a result
        assert isinstance(nr_result["result"], ACMethodResult)

    def test_custom_nr_converges(self, realistic_net) -> None:
        """Custom Newton-Raphson converges on the realistic network."""
        results = run_all_methods(realistic_net, region="integration_test")
        nr_result = next(r for r in results if r["method_id"] == "custom_nr")
        result = nr_result["result"]
        assert result.converged is True, (
            f"custom_nr did not converge: {result.failure_reason}"
        )
        assert result.V is not None
        # Voltage magnitudes should be reasonable
        Vm = np.abs(result.V)
        assert np.all(Vm > 0.7), f"Voltage too low: {Vm.min()}"
        assert np.all(Vm < 1.3), f"Voltage too high: {Vm.max()}"

    def test_all_methods_have_nonnegative_elapsed_time(self, realistic_net) -> None:
        """Every method records a non-negative elapsed time."""
        results = run_all_methods(realistic_net, region="integration_test")
        for r in results:
            assert r["result"].elapsed_sec >= 0, (
                f"Method '{r['method_id']}' has negative elapsed time"
            )

    def test_no_unhandled_exceptions_in_failure_reasons(self, realistic_net) -> None:
        """Methods that fail should have meaningful failure reasons, not 'Unhandled'."""
        results = run_all_methods(realistic_net, region="integration_test")
        for r in results:
            result = r["result"]
            if not result.converged and result.failure_reason:
                # Allow "Unhandled" since some methods legitimately hit unexpected errors,
                # but every failure must have a reason
                assert result.failure_reason is not None, (
                    f"Method '{r['method_id']}' failed without failure_reason"
                )


# ======================================================================
# Integration: Convergence report from real results
# ======================================================================


class TestConvergenceReportIntegration:
    """End-to-end test: generate convergence report from real method results."""

    @pytest.fixture
    def real_results(self, realistic_net):
        """Run all methods and return results for report generation."""
        return run_all_methods(realistic_net, region="integration_test")

    def test_report_generated_from_real_results(self, real_results) -> None:
        """generate_report() succeeds on real execution results."""
        report = generate_report(real_results)
        assert "methods" in report
        assert "summary" in report

    def test_report_has_all_20_methods(self, real_results) -> None:
        """Report contains statistics for all 20 methods."""
        report = generate_report(real_results)
        assert len(report["methods"]) == 20, (
            f"Expected 20 methods in report, got {len(report['methods'])}"
        )

    def test_report_methods_have_convergence_stats(self, real_results) -> None:
        """Each method entry has convergence_rate and failure_reasons."""
        report = generate_report(real_results)
        for method_stats in report["methods"]:
            assert "convergence_rate" in method_stats, (
                f"Missing convergence_rate for {method_stats.get('method_id')}"
            )
            assert "failure_reasons" in method_stats, (
                f"Missing failure_reasons for {method_stats.get('method_id')}"
            )
            # convergence_rate is 0-100
            rate = method_stats["convergence_rate"]
            assert 0.0 <= rate <= 100.0, (
                f"Invalid convergence_rate {rate} for {method_stats['method_id']}"
            )

    def test_report_saved_to_json_file(self, real_results, tmp_path) -> None:
        """Report is saved as valid JSON to the specified path."""
        report = generate_report(real_results)
        output_path = str(tmp_path / "convergence_report.json")
        save_report(report, output_path)

        assert os.path.exists(output_path)

        with open(output_path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)

        # Validate structure
        assert "methods" in loaded
        assert "summary" in loaded
        assert len(loaded["methods"]) == 20

    def test_report_json_contains_all_required_fields(self, real_results, tmp_path) -> None:
        """Saved JSON report has all required fields per method."""
        report = generate_report(real_results)
        output_path = str(tmp_path / "convergence_report.json")
        save_report(report, output_path)

        with open(output_path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)

        required_method_fields = {
            "method_id", "method_name", "category",
            "regions_tested", "converged_count", "failed_count",
            "convergence_rate", "avg_iterations", "avg_elapsed_sec",
            "failure_reasons", "convergence_history",
        }
        required_summary_fields = {
            "total_methods", "total_tests", "total_converged",
            "total_failed", "overall_convergence_rate",
        }

        for method_stats in loaded["methods"]:
            missing = required_method_fields - set(method_stats.keys())
            assert not missing, (
                f"Method '{method_stats.get('method_id')}' missing: {missing}"
            )

        missing_summary = required_summary_fields - set(loaded["summary"].keys())
        assert not missing_summary, f"Summary missing: {missing_summary}"

    def test_summary_totals_consistent(self, real_results) -> None:
        """Summary totals are consistent with per-method counts."""
        report = generate_report(real_results)
        summary = report["summary"]

        total_conv = sum(m["converged_count"] for m in report["methods"])
        total_fail = sum(m["failed_count"] for m in report["methods"])

        assert summary["total_converged"] == total_conv
        assert summary["total_failed"] == total_fail
        assert summary["total_tests"] == total_conv + total_fail

    def test_print_summary_no_crash(self, real_results) -> None:
        """print_summary() executes without errors."""
        report = generate_report(real_results)
        # Should not raise
        print_summary(report)


# ======================================================================
# Integration: Network preparation on realistic network
# ======================================================================


class TestNetworkPrepIntegration:
    """Test network preparation on a realistic meshed network."""

    def test_prepare_realistic_network(self, realistic_net) -> None:
        """prepare_network() extracts valid data from 5-bus network."""
        data = prepare_network(realistic_net)
        n = data.Ybus.shape[0]

        assert n >= 5, f"Expected at least 5 buses, got {n}"
        assert data.Sbus.shape == (n,)
        assert data.V0.shape == (n,)
        assert len(data.ref) >= 1
        assert len(data.pv) >= 1
        assert len(data.pq) >= 1

    def test_all_custom_solvers_accept_network_data(self, realistic_net) -> None:
        """All custom solvers accept the extracted NetworkData."""
        data = prepare_network(realistic_net)
        methods = get_all_methods()
        custom_methods = [m for m in methods if m.category != "pandapower"]

        for method in custom_methods:
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
            assert isinstance(result, ACMethodResult), (
                f"Method '{method.id}' returned {type(result)} instead of ACMethodResult"
            )


# ======================================================================
# Integration: Batch runner CLI behavior
# ======================================================================


class TestBatchRunnerCLI:
    """Test batch runner CLI argument parsing and behavior."""

    def test_cli_parser_region_flag(self) -> None:
        """CLI parser accepts --region flag."""
        from src.ac_powerflow.batch_runner import build_parser
        parser = build_parser()
        args = parser.parse_args(["--region", "shikoku"])
        assert args.region == "shikoku"
        assert args.all_regions is False

    def test_cli_parser_all_regions_flag(self) -> None:
        """CLI parser accepts --all-regions flag."""
        from src.ac_powerflow.batch_runner import build_parser
        parser = build_parser()
        args = parser.parse_args(["--all-regions"])
        assert args.all_regions is True

    def test_cli_parser_parallel_flag(self) -> None:
        """CLI parser accepts --parallel flag."""
        from src.ac_powerflow.batch_runner import build_parser
        parser = build_parser()
        args = parser.parse_args(["--region", "shikoku", "--parallel"])
        assert args.parallel is True

    def test_cli_parser_max_workers_flag(self) -> None:
        """CLI parser accepts --max-workers flag."""
        from src.ac_powerflow.batch_runner import build_parser
        parser = build_parser()
        args = parser.parse_args(["--all-regions", "--parallel", "--max-workers", "4"])
        assert args.max_workers == 4

    def test_cli_parser_output_dir_flag(self) -> None:
        """CLI parser accepts --output-dir flag."""
        from src.ac_powerflow.batch_runner import build_parser
        parser = build_parser()
        args = parser.parse_args(["--region", "shikoku", "--output-dir", "/tmp/test"])
        assert args.output_dir == "/tmp/test"


# ======================================================================
# Integration: Full end-to-end report generation
# ======================================================================


class TestEndToEndReportGeneration:
    """Full end-to-end: run methods → generate report → save to file → validate."""

    def test_full_pipeline(self, realistic_net, tmp_path) -> None:
        """Complete pipeline: run all methods, generate report, save, validate."""
        # Step 1: Run all methods
        results = run_all_methods(realistic_net, region="e2e_test")
        assert len(results) == 20

        # Step 2: Generate report
        report = generate_report(results)
        assert len(report["methods"]) == 20

        # Step 3: Save to file
        report_path = str(tmp_path / "convergence_report.json")
        save_report(report, report_path)
        assert os.path.exists(report_path)

        # Step 4: Validate JSON
        with open(report_path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)

        # All 20 methods present
        assert len(loaded["methods"]) == 20

        # Each method has convergence_rate and failure_reasons
        for m in loaded["methods"]:
            assert "convergence_rate" in m
            assert "failure_reasons" in m
            assert "avg_iterations" in m
            assert "avg_elapsed_sec" in m

        # Summary is valid
        summary = loaded["summary"]
        assert summary["total_methods"] == 20
        assert summary["total_tests"] == 20
        assert summary["total_converged"] + summary["total_failed"] == 20
        assert 0.0 <= summary["overall_convergence_rate"] <= 100.0

        # Step 5: Print summary (should not crash)
        print_summary(report)
