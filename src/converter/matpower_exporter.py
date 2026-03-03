"""Export pandapower networks to MATPOWER .mat format.

Converts pandapower network models into MATPOWER case files using
pandapower's ``to_mpc()`` converter.  Supports per-region and national
(merged) model export.  Generates a validation report summarising bus,
branch, and generator table sizes, power flow feasibility, and any
warnings encountered during export.

Output file layout::

    output/matpower/
        japan_grid_all.mat          # National model
        regions/
            hokkaido.mat            # Per-region models
            tohoku.mat
            ...
    output/reports/
        validation_report.json      # Export validation report

Usage::

    from src.converter.matpower_exporter import MATPOWERExporter

    exporter = MATPOWERExporter()
    result = exporter.export_region(net, region="shikoku")
    # result.mat_path  -> Path to the generated .mat file
    # result.report    -> Validation report dict
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.io import savemat

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# Default output directories (relative to project root)
_DEFAULT_MATPOWER_DIR = "output/matpower"
_DEFAULT_REGIONS_DIR = "output/matpower/regions"
_DEFAULT_REPORTS_DIR = "output/reports"
_DEFAULT_REPORT_FILENAME = "validation_report.json"


@dataclass
class ExportResult:
    """Result of a single MATPOWER export operation.

    Attributes:
        region: Region identifier for the exported network.
        mat_path: Filesystem path to the generated .mat file.
        bus_count: Number of buses in the MATPOWER case.
        branch_count: Number of branches (lines + transformers).
        gen_count: Number of generators.
        base_mva: System base MVA.
        success: Whether the export completed without errors.
        warnings: List of warning messages from the export process.
        report: Validation report dict for this export.
    """

    region: str
    mat_path: str = ""
    bus_count: int = 0
    branch_count: int = 0
    gen_count: int = 0
    base_mva: float = 100.0
    success: bool = False
    warnings: List[str] = field(default_factory=list)
    report: Dict[str, object] = field(default_factory=dict)

    @property
    def summary(self) -> Dict[str, object]:
        """Return a summary dict for logging."""
        return {
            "region": self.region,
            "buses": self.bus_count,
            "branches": self.branch_count,
            "generators": self.gen_count,
            "base_mva": self.base_mva,
            "success": self.success,
            "warnings": len(self.warnings),
        }


class MATPOWERExporter:
    """Exports pandapower networks to MATPOWER .mat case files.

    Wraps pandapower's ``to_mpc()`` converter with project-specific
    directory layout, file naming, and validation reporting.

    Args:
        output_dir: Root output directory for .mat files.
            Defaults to ``output/matpower``.
        reports_dir: Directory for validation reports.
            Defaults to ``output/reports``.
    """

    def __init__(
        self,
        output_dir: str = _DEFAULT_MATPOWER_DIR,
        reports_dir: str = _DEFAULT_REPORTS_DIR,
    ) -> None:
        self._output_dir = output_dir
        self._regions_dir = os.path.join(output_dir, "regions")
        self._reports_dir = reports_dir

    # ------------------------------------------------------------------
    # Public API: per-region export
    # ------------------------------------------------------------------

    def export_region(
        self,
        net: Any,
        region: str,
        *,
        run_powerflow: bool = False,
    ) -> ExportResult:
        """Export a single-region pandapower network to MATPOWER .mat.

        Args:
            net: The pandapower network to export.
            region: Region identifier (e.g., 'shikoku', 'hokkaido').
            run_powerflow: If True, attempt ``pp.runpp()`` before export
                to populate voltage and power flow results.  Convergence
                failure is logged as a warning but does not abort export.

        Returns:
            ExportResult with export metadata and validation report.
        """
        result = ExportResult(region=region)

        # Optionally run power flow
        if run_powerflow:
            self._try_powerflow(net, result)

        # Convert and save
        mat_path = os.path.join(self._regions_dir, f"{region}.mat")
        self._export_to_mat(net, mat_path, result)

        # Build validation report
        result.report = self._build_report(result)

        logger.info(
            "Exported region '%s' to MATPOWER: %s",
            region,
            result.summary,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: national model export
    # ------------------------------------------------------------------

    def export_national(
        self,
        net: Any,
        *,
        run_powerflow: bool = False,
    ) -> ExportResult:
        """Export the national (merged) pandapower network to MATPOWER .mat.

        Args:
            net: The national pandapower network.
            run_powerflow: If True, attempt power flow before export.

        Returns:
            ExportResult with export metadata and validation report.
        """
        result = ExportResult(region="national")

        if run_powerflow:
            self._try_powerflow(net, result)

        mat_path = os.path.join(self._output_dir, "japan_grid_all.mat")
        self._export_to_mat(net, mat_path, result)

        result.report = self._build_report(result)

        logger.info(
            "Exported national model to MATPOWER: %s",
            result.summary,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: batch export (multiple regions + national)
    # ------------------------------------------------------------------

    def export_all(
        self,
        region_nets: Dict[str, Any],
        national_net: Optional[Any] = None,
        *,
        run_powerflow: bool = False,
    ) -> Dict[str, ExportResult]:
        """Export multiple regional networks and optional national model.

        Args:
            region_nets: Mapping of region name to pandapower network.
            national_net: Optional national (merged) pandapower network.
            run_powerflow: If True, attempt power flow before each export.

        Returns:
            Dict mapping region name (and 'national') to ExportResult.
        """
        results: Dict[str, ExportResult] = {}

        for region, net in region_nets.items():
            results[region] = self.export_region(
                net, region, run_powerflow=run_powerflow,
            )

        if national_net is not None:
            results["national"] = self.export_national(
                national_net, run_powerflow=run_powerflow,
            )

        # Write combined validation report
        self._write_combined_report(results)

        logger.info(
            "Batch export complete: %d regions, national=%s",
            len(region_nets),
            national_net is not None,
        )

        return results

    # ------------------------------------------------------------------
    # Public API: validation report generation
    # ------------------------------------------------------------------

    def write_validation_report(
        self,
        results: Dict[str, ExportResult],
        report_path: Optional[str] = None,
    ) -> str:
        """Write a combined validation report to JSON.

        Args:
            results: Mapping of region to ExportResult.
            report_path: Override path for the report file.

        Returns:
            Path to the written report file.
        """
        return self._write_combined_report(results, report_path)

    # ------------------------------------------------------------------
    # Internal: MATPOWER conversion and file writing
    # ------------------------------------------------------------------

    def _export_to_mat(
        self,
        net: Any,
        mat_path: str,
        result: ExportResult,
    ) -> None:
        """Convert pandapower network to MPC dict and save as .mat file.

        Uses pandapower's ``to_mpc()`` for the conversion from
        pandapower to MATPOWER format (0-based → 1-based indexing),
        then ``scipy.io.savemat()`` for file output.

        Args:
            net: The pandapower network.
            mat_path: Output file path for the .mat file.
            result: ExportResult to update with counts and status.
        """
        try:
            # Import here to keep the module importable even when
            # pandapower is not installed (graceful degradation).
            try:
                from pandapower.converter.matpower.to_mpc import to_mpc
            except ImportError:
                from pandapower.converter import to_mpc

            mpc = to_mpc(net, init="flat")

        except Exception as exc:
            msg = f"Failed to convert to MATPOWER format: {exc}"
            result.warnings.append(msg)
            logger.error(msg)
            result.success = False
            return

        # Extract table dimensions from the MPC dict
        mpc_data = mpc.get("mpc", mpc)  # Handle both wrapped and unwrapped
        self._extract_mpc_stats(mpc_data, result)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(mat_path), exist_ok=True)

        try:
            savemat(mat_path, mpc)
            result.mat_path = mat_path
            result.success = True

            logger.info(
                "Saved MATPOWER file: %s (buses=%d, branches=%d, gens=%d)",
                mat_path,
                result.bus_count,
                result.branch_count,
                result.gen_count,
            )

        except Exception as exc:
            msg = f"Failed to save .mat file '{mat_path}': {exc}"
            result.warnings.append(msg)
            logger.error(msg)
            result.success = False

    def _extract_mpc_stats(
        self,
        mpc_data: Dict[str, Any],
        result: ExportResult,
    ) -> None:
        """Extract bus/branch/gen counts and baseMVA from an MPC dict.

        Args:
            mpc_data: The inner MATPOWER case dict.
            result: ExportResult to populate.
        """
        bus = mpc_data.get("bus")
        if bus is not None:
            result.bus_count = int(np.size(bus, 0)) if hasattr(bus, "shape") else 0

        branch = mpc_data.get("branch")
        if branch is not None:
            result.branch_count = int(np.size(branch, 0)) if hasattr(branch, "shape") else 0

        gen = mpc_data.get("gen")
        if gen is not None:
            result.gen_count = int(np.size(gen, 0)) if hasattr(gen, "shape") else 0

        base_mva = mpc_data.get("baseMVA")
        if base_mva is not None:
            result.base_mva = float(np.squeeze(base_mva))

    # ------------------------------------------------------------------
    # Internal: optional power flow
    # ------------------------------------------------------------------

    def _try_powerflow(self, net: Any, result: ExportResult) -> None:
        """Attempt to run pandapower power flow before export.

        Power flow convergence failure is non-fatal — the network can
        still be exported to MATPOWER without solved voltages.

        Args:
            net: The pandapower network.
            result: ExportResult to append warnings to.
        """
        try:
            import pandapower as pp

            pp.runpp(net)
            logger.info(
                "Power flow converged for '%s'",
                result.region,
            )

        except Exception as exc:
            msg = (
                f"Power flow did not converge for '{result.region}': "
                f"{exc}"
            )
            result.warnings.append(msg)
            logger.warning(msg)

    # ------------------------------------------------------------------
    # Internal: validation report
    # ------------------------------------------------------------------

    def _build_report(self, result: ExportResult) -> Dict[str, object]:
        """Build a validation report dict for a single export.

        Args:
            result: The ExportResult to report on.

        Returns:
            Dict containing validation results.
        """
        report: Dict[str, object] = {
            "region": result.region,
            "export_success": result.success,
            "mat_path": result.mat_path,
            "tables": {
                "bus_count": result.bus_count,
                "branch_count": result.branch_count,
                "gen_count": result.gen_count,
                "base_mva": result.base_mva,
            },
            "validation": {
                "has_buses": result.bus_count > 0,
                "has_branches": result.branch_count > 0,
                "has_generators": result.gen_count > 0,
                "tables_non_empty": (
                    result.bus_count > 0
                    and result.branch_count > 0
                ),
            },
            "warnings": list(result.warnings),
            "warning_count": len(result.warnings),
        }

        return report

    def _write_combined_report(
        self,
        results: Dict[str, ExportResult],
        report_path: Optional[str] = None,
    ) -> str:
        """Write a combined validation report for all exported regions.

        Args:
            results: Mapping of region name to ExportResult.
            report_path: Override path; defaults to
                ``output/reports/validation_report.json``.

        Returns:
            Path to the written report file.
        """
        if report_path is None:
            report_path = os.path.join(
                self._reports_dir, _DEFAULT_REPORT_FILENAME,
            )

        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        # Aggregate per-region reports
        region_reports = {}
        total_buses = 0
        total_branches = 0
        total_gens = 0
        all_warnings: List[str] = []
        success_count = 0

        for region, res in results.items():
            region_reports[region] = res.report if res.report else self._build_report(res)
            total_buses += res.bus_count
            total_branches += res.branch_count
            total_gens += res.gen_count
            all_warnings.extend(res.warnings)
            if res.success:
                success_count += 1

        combined_report: Dict[str, object] = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_regions_exported": len(results),
                "successful_exports": success_count,
                "failed_exports": len(results) - success_count,
            },
            "summary": {
                "total_buses": total_buses,
                "total_branches": total_branches,
                "total_generators": total_gens,
                "total_warnings": len(all_warnings),
            },
            "regions": region_reports,
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(combined_report, f, ensure_ascii=False, indent=2)

        logger.info(
            "Validation report written to %s (%d regions, %d warnings)",
            report_path,
            len(results),
            len(all_warnings),
        )

        return report_path
