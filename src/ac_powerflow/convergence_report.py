"""Generate convergence analysis reports for AC power flow methods.

Produces per-method convergence statistics from batch execution results,
writes a structured JSON report to ``output/ac_powerflow/``, and prints
a console summary table.

Usage::

    from src.ac_powerflow.convergence_report import (
        generate_report,
        save_report,
        print_summary,
    )

    report = generate_report(results)
    save_report(report, "output/ac_powerflow/convergence_report.json")
    print_summary(report)
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from src.ac_powerflow.solver_interface import ACMethodResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = "output/ac_powerflow"
DEFAULT_REPORT_FILENAME = "convergence_report.json"


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def generate_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a convergence analysis report from batch execution results.

    Aggregates per-method statistics across all regions tested, including
    convergence rates, average iterations, timing, failure reasons, and
    convergence history trajectories.

    Args:
        results: List of result records.  Each record is a dict with keys:

            * ``method_id`` (str) — Unique method identifier.
            * ``method_name`` (str) — Human-readable method name.
            * ``category`` (str) — Method category.
            * ``region`` (str) — Region identifier.
            * ``result`` (:class:`ACMethodResult`) — Solver result object.

    Returns:
        Report dict with keys ``methods`` (list of per-method statistics)
        and ``summary`` (overall statistics).
    """
    # Group results by method_id
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in results:
        method_id = record.get("method_id", "unknown")
        grouped[method_id].append(record)

    method_stats: List[Dict[str, Any]] = []
    total_converged = 0
    total_failed = 0
    total_tests = 0

    for method_id, records in sorted(grouped.items()):
        stats = _compute_method_stats(method_id, records)
        method_stats.append(stats)
        total_converged += stats["converged_count"]
        total_failed += stats["failed_count"]
        total_tests += stats["regions_tested"]

    report: Dict[str, Any] = {
        "methods": method_stats,
        "summary": {
            "total_methods": len(method_stats),
            "total_tests": total_tests,
            "total_converged": total_converged,
            "total_failed": total_failed,
            "overall_convergence_rate": (
                round(total_converged / total_tests * 100, 2)
                if total_tests > 0
                else 0.0
            ),
        },
    }

    logger.info(
        "Generated convergence report: %d methods, %d tests, %.1f%% overall rate",
        len(method_stats),
        total_tests,
        report["summary"]["overall_convergence_rate"],
    )

    return report


def save_report(
    report: Dict[str, Any],
    output_path: str = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_REPORT_FILENAME),
) -> str:
    """Write convergence report to a JSON file.

    Args:
        report: Report dict produced by :func:`generate_report`.
        output_path: Destination file path.  Parent directories are
            created automatically.

    Returns:
        Path to the written JSON file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    logger.info("Saved convergence report: %s", output_path)
    return output_path


def print_summary(report: Dict[str, Any]) -> None:
    """Print a console table summarising per-method convergence rates.

    Args:
        report: Report dict produced by :func:`generate_report`.
    """
    methods = report.get("methods", [])
    summary = report.get("summary", {})

    if not methods:
        logger.warning("No methods in report — nothing to print.")
        return

    # Header
    header = (
        f"{'Method':<30} {'Category':<20} {'Regions':>7} "
        f"{'Conv':>5} {'Fail':>5} {'Rate':>7} "
        f"{'Avg Iter':>9} {'Avg Time':>9}"
    )
    separator = "-" * len(header)

    lines: List[str] = []
    lines.append("")
    lines.append("=" * len(header))
    lines.append("  AC Power Flow Convergence Summary")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(separator)

    current_category = None
    for method in methods:
        category = method.get("category", "")
        if category != current_category:
            if current_category is not None:
                lines.append(separator)
            current_category = category

        method_name = method.get("method_name", method.get("method_id", "?"))
        rate_str = f"{method['convergence_rate']:.1f}%"
        avg_iter_str = f"{method['avg_iterations']:.1f}"
        avg_time_str = f"{method['avg_elapsed_sec']:.4f}s"

        lines.append(
            f"{method_name:<30} {category:<20} {method['regions_tested']:>7} "
            f"{method['converged_count']:>5} {method['failed_count']:>5} "
            f"{rate_str:>7} {avg_iter_str:>9} {avg_time_str:>9}"
        )

    lines.append(separator)

    # Overall summary line
    overall_rate = summary.get("overall_convergence_rate", 0.0)
    lines.append(
        f"{'TOTAL':<30} {'':<20} {summary.get('total_tests', 0):>7} "
        f"{summary.get('total_converged', 0):>5} "
        f"{summary.get('total_failed', 0):>5} "
        f"{overall_rate:.1f}%"
    )
    lines.append("=" * len(header))
    lines.append("")

    output = "\n".join(lines)
    print(output)
    logger.info("Printed convergence summary for %d methods.", len(methods))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _compute_method_stats(
    method_id: str,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute convergence statistics for a single method across regions.

    Args:
        method_id: Unique method identifier.
        records: List of result records for this method.

    Returns:
        Dict with per-method statistics.
    """
    # Extract metadata from the first record
    first = records[0]
    method_name = first.get("method_name", method_id)
    category = first.get("category", "unknown")

    regions_tested = len(records)
    converged_count = 0
    failed_count = 0
    all_iterations: List[int] = []
    all_elapsed: List[float] = []
    failure_reasons: Dict[str, int] = defaultdict(int)
    convergence_histories: Dict[str, List[float]] = {}

    for record in records:
        result: ACMethodResult = record["result"]
        region = record.get("region", "unknown")

        if result.converged:
            converged_count += 1
        else:
            failed_count += 1
            reason = result.failure_reason or "Unknown failure"
            failure_reasons[reason] += 1

        all_iterations.append(result.iterations)
        all_elapsed.append(result.elapsed_sec)

        # Store convergence history per region
        if result.convergence_history:
            convergence_histories[region] = list(result.convergence_history)

    convergence_rate = (
        round(converged_count / regions_tested * 100, 2)
        if regions_tested > 0
        else 0.0
    )
    avg_iterations = (
        round(sum(all_iterations) / len(all_iterations), 2)
        if all_iterations
        else 0.0
    )
    avg_elapsed_sec = (
        round(sum(all_elapsed) / len(all_elapsed), 6)
        if all_elapsed
        else 0.0
    )

    # Convert failure_reasons defaultdict to sorted list of dicts
    failure_reasons_list = [
        {"reason": reason, "count": count}
        for reason, count in sorted(
            failure_reasons.items(), key=lambda x: -x[1]
        )
    ]

    return {
        "method_id": method_id,
        "method_name": method_name,
        "category": category,
        "regions_tested": regions_tested,
        "converged_count": converged_count,
        "failed_count": failed_count,
        "convergence_rate": convergence_rate,
        "avg_iterations": avg_iterations,
        "avg_elapsed_sec": avg_elapsed_sec,
        "failure_reasons": failure_reasons_list,
        "convergence_history": convergence_histories,
    }
