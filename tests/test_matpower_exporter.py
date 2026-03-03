"""Unit tests for the MATPOWERExporter module.

Tests MATPOWER .mat file export from pandapower networks, including
per-region export, national model export, batch export, validation
report generation, power flow attempt handling, and the ExportResult
dataclass.
"""

import json
import os
from pathlib import Path

import pandapower as pp
import pytest
from scipy.io import loadmat

from src.converter.matpower_exporter import (
    ExportResult,
    MATPOWERExporter,
)
from src.converter.pandapower_builder import PandapowerBuilder
from src.model.grid_network import GridNetwork


# ======================================================================
# Helper: build a pandapower network from a GridNetwork fixture
# ======================================================================


def _build_pp_net(network: GridNetwork):
    """Build a pandapower network from a GridNetwork using PandapowerBuilder."""
    builder = PandapowerBuilder()
    result = builder.build(network)
    return result.net


# ======================================================================
# ExportResult dataclass
# ======================================================================


class TestExportResult:
    """Tests for the ExportResult dataclass."""

    def test_default_values(self) -> None:
        """Default ExportResult fields are correct."""
        result = ExportResult(region="test")
        assert result.region == "test"
        assert result.mat_path == ""
        assert result.bus_count == 0
        assert result.branch_count == 0
        assert result.gen_count == 0
        assert result.base_mva == 100.0
        assert result.success is False
        assert result.warnings == []
        assert result.report == {}

    def test_summary_keys(self) -> None:
        """Summary dict contains expected keys."""
        result = ExportResult(
            region="shikoku",
            bus_count=10,
            branch_count=8,
            gen_count=3,
            success=True,
        )
        summary = result.summary
        assert summary["region"] == "shikoku"
        assert summary["buses"] == 10
        assert summary["branches"] == 8
        assert summary["generators"] == 3
        assert summary["success"] is True


# ======================================================================
# MATPOWERExporter: initialization
# ======================================================================


class TestMATPOWERExporterInit:
    """Tests for MATPOWERExporter construction."""

    def test_default_dirs(self) -> None:
        """Default output directories are set."""
        exporter = MATPOWERExporter()
        assert "output/matpower" in exporter._output_dir
        assert "output/reports" in exporter._reports_dir

    def test_custom_dirs(self, tmp_path: Path) -> None:
        """Custom output directories are accepted."""
        mat_dir = str(tmp_path / "custom_mat")
        rep_dir = str(tmp_path / "custom_reports")
        exporter = MATPOWERExporter(output_dir=mat_dir, reports_dir=rep_dir)
        assert exporter._output_dir == mat_dir
        assert exporter._reports_dir == rep_dir


# ======================================================================
# MATPOWERExporter: export_region
# ======================================================================


class TestExportRegion:
    """Tests for per-region MATPOWER export."""

    def test_creates_mat_file(
        self,
        sample_grid_network: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """export_region creates a .mat file."""
        net = _build_pp_net(sample_grid_network)
        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        result = exporter.export_region(net, "shikoku")
        assert result.success is True
        assert os.path.exists(result.mat_path)
        assert result.mat_path.endswith(".mat")

    def test_mat_file_loadable(
        self,
        sample_grid_network: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """.mat file is loadable by scipy."""
        net = _build_pp_net(sample_grid_network)
        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        result = exporter.export_region(net, "shikoku")
        data = loadmat(result.mat_path)
        assert "mpc" in data

    def test_bus_count_populated(
        self,
        sample_grid_network: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """ExportResult has non-zero bus_count."""
        net = _build_pp_net(sample_grid_network)
        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        result = exporter.export_region(net, "shikoku")
        assert result.bus_count > 0
        # MPC conversion may drop isolated buses (no branches connected)
        assert result.bus_count <= len(net.bus)

    def test_branch_count_populated(
        self,
        sample_grid_network: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """ExportResult has non-zero branch_count."""
        net = _build_pp_net(sample_grid_network)
        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        result = exporter.export_region(net, "shikoku")
        assert result.branch_count > 0

    def test_gen_count_populated(
        self,
        sample_grid_network: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """ExportResult has correct gen_count (gens + ext_grid)."""
        net = _build_pp_net(sample_grid_network)
        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        result = exporter.export_region(net, "shikoku")
        # gen_count includes both generators and ext_grid elements
        assert result.gen_count > 0

    def test_report_populated(
        self,
        sample_grid_network: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """ExportResult.report dict is populated after export."""
        net = _build_pp_net(sample_grid_network)
        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        result = exporter.export_region(net, "shikoku")
        assert result.report is not None
        assert result.report["region"] == "shikoku"
        assert result.report["export_success"] is True
        assert "tables" in result.report
        assert "validation" in result.report

    def test_report_validation_flags(
        self,
        sample_grid_network: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """Validation section in report has correct boolean flags."""
        net = _build_pp_net(sample_grid_network)
        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        result = exporter.export_region(net, "shikoku")
        validation = result.report["validation"]
        assert validation["has_buses"] is True
        assert validation["has_branches"] is True
        assert validation["tables_non_empty"] is True


# ======================================================================
# MATPOWERExporter: export_national
# ======================================================================


class TestExportNational:
    """Tests for national model MATPOWER export."""

    def test_creates_national_mat_file(
        self,
        sample_grid_network: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """export_national creates a national .mat file."""
        net = _build_pp_net(sample_grid_network)
        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        result = exporter.export_national(net)
        assert result.success is True
        assert result.region == "national"
        assert os.path.exists(result.mat_path)
        assert "japan_grid_all" in result.mat_path


# ======================================================================
# MATPOWERExporter: export_all (batch)
# ======================================================================


class TestExportAll:
    """Tests for batch MATPOWER export."""

    def test_export_multiple_regions(
        self,
        sample_grid_network: GridNetwork,
        sample_grid_network_50hz: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """export_all creates .mat files for each region."""
        net_shikoku = _build_pp_net(sample_grid_network)
        net_hokkaido = _build_pp_net(sample_grid_network_50hz)

        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        results = exporter.export_all(
            region_nets={"shikoku": net_shikoku, "hokkaido": net_hokkaido},
        )

        assert "shikoku" in results
        assert "hokkaido" in results
        assert results["shikoku"].success is True
        assert results["hokkaido"].success is True

    def test_export_all_with_national(
        self,
        sample_grid_network: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """export_all includes national model when provided."""
        net = _build_pp_net(sample_grid_network)

        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        results = exporter.export_all(
            region_nets={"shikoku": net},
            national_net=net,
        )

        assert "shikoku" in results
        assert "national" in results
        assert results["national"].success is True

    def test_combined_report_written(
        self,
        sample_grid_network: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """export_all writes a combined validation report JSON."""
        net = _build_pp_net(sample_grid_network)

        reports_dir = str(tmp_path / "reports")
        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=reports_dir,
        )
        exporter.export_all(region_nets={"shikoku": net})

        report_path = os.path.join(reports_dir, "validation_report.json")
        assert os.path.exists(report_path)

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert "metadata" in report
        assert "summary" in report
        assert "regions" in report
        assert report["metadata"]["total_regions_exported"] == 1
        assert report["metadata"]["successful_exports"] == 1


# ======================================================================
# MATPOWERExporter: write_validation_report
# ======================================================================


class TestWriteValidationReport:
    """Tests for standalone validation report writing."""

    def test_write_custom_path(
        self,
        sample_grid_network: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """write_validation_report accepts a custom path."""
        net = _build_pp_net(sample_grid_network)
        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        result = exporter.export_region(net, "shikoku")

        custom_path = str(tmp_path / "custom_report.json")
        written_path = exporter.write_validation_report(
            {"shikoku": result},
            report_path=custom_path,
        )
        assert os.path.exists(written_path)
        assert written_path == custom_path

    def test_report_aggregates_statistics(
        self,
        sample_grid_network: GridNetwork,
        sample_grid_network_50hz: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """Combined report aggregates bus/branch/gen totals."""
        net_s = _build_pp_net(sample_grid_network)
        net_h = _build_pp_net(sample_grid_network_50hz)

        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        result_s = exporter.export_region(net_s, "shikoku")
        result_h = exporter.export_region(net_h, "hokkaido")

        report_path = str(tmp_path / "agg_report.json")
        exporter.write_validation_report(
            {"shikoku": result_s, "hokkaido": result_h},
            report_path=report_path,
        )

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        total_buses = report["summary"]["total_buses"]
        assert total_buses == result_s.bus_count + result_h.bus_count

        total_branches = report["summary"]["total_branches"]
        assert total_branches == result_s.branch_count + result_h.branch_count


# ======================================================================
# MATPOWERExporter: edge cases
# ======================================================================


class TestMatpowerEdgeCases:
    """Tests for edge cases in MATPOWER export."""

    def test_empty_network_export(self, tmp_path: Path) -> None:
        """Network with only buses (no lines/gens) exports successfully."""
        net = pp.create_empty_network(f_hz=60)
        pp.create_bus(net, vn_kv=275.0, name="TestBus")
        pp.create_ext_grid(net, bus=0, vm_pu=1.0)

        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        result = exporter.export_region(net, "test")
        assert result.success is True
        assert result.bus_count == 1
        assert result.branch_count == 0

    def test_directory_creation(self, tmp_path: Path) -> None:
        """Export creates missing directories automatically."""
        net = pp.create_empty_network(f_hz=60)
        pp.create_bus(net, vn_kv=275.0, name="TestBus")
        pp.create_ext_grid(net, bus=0, vm_pu=1.0)

        deep_dir = str(tmp_path / "deep" / "nested" / "matpower")
        exporter = MATPOWERExporter(
            output_dir=deep_dir,
            reports_dir=str(tmp_path / "reports"),
        )
        result = exporter.export_region(net, "test")
        assert result.success is True
        assert os.path.exists(result.mat_path)

    def test_base_mva_extracted(
        self,
        sample_grid_network: GridNetwork,
        tmp_path: Path,
    ) -> None:
        """base_mva is extracted from the MATPOWER case data."""
        net = _build_pp_net(sample_grid_network)
        exporter = MATPOWERExporter(
            output_dir=str(tmp_path / "matpower"),
            reports_dir=str(tmp_path / "reports"),
        )
        result = exporter.export_region(net, "shikoku")
        # pandapower default baseMVA is 1.0 (not 100 like standard MATPOWER)
        # but the exact value depends on to_mpc() internals
        assert isinstance(result.base_mva, float)
        assert result.base_mva > 0
