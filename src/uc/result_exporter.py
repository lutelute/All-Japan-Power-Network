"""Export UC solver results to XML and CSV formats.

Serializes :class:`~src.uc.models.UCResult` data to:

- **XML** conforming to ``schemas/uc_result.xsd`` (namespace ``urn:japan-grid:v1``).
- **CSV** tabular output with one row per (generator, period) pair.

Follows the lxml patterns established in
:mod:`src.standardizer.xml_exporter` (namespace constants,
``_format_decimal()``, ``etree.Element`` builders).

Usage::

    from src.uc.models import UCResult
    from src.uc.result_exporter import export_uc_result_xml, export_uc_result_csv

    result: UCResult = ...  # from solver
    export_uc_result_xml(result, "output/uc_result.xml")
    export_uc_result_csv(result, "output/uc_result.csv")
"""

import csv
import os
from decimal import Decimal
from typing import Optional

from lxml import etree

from src.uc.models import GeneratorSchedule, UCResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# XML namespace for the Japan Grid schema (same as power_grid.xsd)
NAMESPACE = "urn:japan-grid:v1"
XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"


def _format_decimal(value: float) -> str:
    """Format a float to a decimal string without unnecessary trailing zeros.

    Uses Decimal for precise representation, which is required by the XSD
    NonNegativeDecimalType.

    Args:
        value: Numeric value to format.

    Returns:
        String representation of the decimal value.
    """
    d = Decimal(str(value))
    normalized = d.normalize()
    return format(normalized, "f")


def _ensure_output_dir(output_path: str) -> None:
    """Ensure the output directory exists, creating it if necessary.

    Args:
        output_path: File path for the output.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------------
# XML Export
# ------------------------------------------------------------------


def _build_period_element(
    t: int, commitment: int, power_output_mw: float
) -> etree._Element:
    """Build a Period XML element for a single timestep.

    Args:
        t: Period index.
        commitment: Binary commitment status (0 or 1).
        power_output_mw: Power output in MW for this period.

    Returns:
        lxml Element representing the Period.
    """
    attrs = {
        "t": str(t),
        "commitment": str(commitment),
        "power_output_mw": _format_decimal(power_output_mw),
    }
    return etree.Element(f"{{{NAMESPACE}}}Period", **attrs)


def _build_generator_cost_element(
    schedule: GeneratorSchedule,
) -> etree._Element:
    """Build a Cost XML element for a generator's cost breakdown.

    Only includes cost attributes when they are non-zero.

    Args:
        schedule: GeneratorSchedule with cost breakdown fields.

    Returns:
        lxml Element representing the generator Cost.
    """
    attrs = {}
    if schedule.fuel_cost != 0.0:
        attrs["fuel_cost"] = _format_decimal(schedule.fuel_cost)
    if schedule.startup_cost != 0.0:
        attrs["startup_cost"] = _format_decimal(schedule.startup_cost)
    if schedule.shutdown_cost != 0.0:
        attrs["shutdown_cost"] = _format_decimal(schedule.shutdown_cost)
    if schedule.no_load_cost != 0.0:
        attrs["no_load_cost"] = _format_decimal(schedule.no_load_cost)
    return etree.Element(f"{{{NAMESPACE}}}Cost", **attrs)


def _build_generator_schedule_element(
    schedule: GeneratorSchedule,
) -> etree._Element:
    """Build a GeneratorSchedule XML element.

    Includes Period sub-elements for each timestep and a Cost sub-element
    summarising the generator's cost breakdown.

    Args:
        schedule: GeneratorSchedule dataclass instance.

    Returns:
        lxml Element representing the GeneratorSchedule.
    """
    attrs = {
        "generator_id": schedule.generator_id,
    }
    if schedule.total_cost != 0.0:
        attrs["total_cost"] = _format_decimal(schedule.total_cost)
    attrs["num_startups"] = str(schedule.num_startups)
    attrs["capacity_factor"] = _format_decimal(schedule.capacity_factor)

    elem = etree.Element(f"{{{NAMESPACE}}}GeneratorSchedule", **attrs)

    # Period elements
    for t, (commitment, power_mw) in enumerate(
        zip(schedule.commitment, schedule.power_output_mw)
    ):
        period_elem = _build_period_element(t, commitment, power_mw)
        elem.append(period_elem)

    # Cost breakdown element
    cost_elem = _build_generator_cost_element(schedule)
    elem.append(cost_elem)

    return elem


def _build_cost_breakdown_element(result: UCResult) -> etree._Element:
    """Build a system-wide CostBreakdown XML element.

    Aggregates cost components across all generator schedules.

    Args:
        result: UCResult with generator schedules.

    Returns:
        lxml Element representing the system CostBreakdown.
    """
    total_fuel = sum(s.fuel_cost for s in result.schedules)
    total_startup = sum(s.startup_cost for s in result.schedules)
    total_shutdown = sum(s.shutdown_cost for s in result.schedules)
    total_no_load = sum(s.no_load_cost for s in result.schedules)

    attrs = {}
    if total_fuel != 0.0:
        attrs["fuel_cost"] = _format_decimal(total_fuel)
    if total_startup != 0.0:
        attrs["startup_cost"] = _format_decimal(total_startup)
    if total_shutdown != 0.0:
        attrs["shutdown_cost"] = _format_decimal(total_shutdown)
    if total_no_load != 0.0:
        attrs["no_load_cost"] = _format_decimal(total_no_load)

    return etree.Element(f"{{{NAMESPACE}}}CostBreakdown", **attrs)


def _build_diagnostics_element(result: UCResult) -> Optional[etree._Element]:
    """Build a Diagnostics XML element containing warnings.

    Returns ``None`` if there are no diagnostics to report.

    Args:
        result: UCResult with optional warnings.

    Returns:
        lxml Element representing Diagnostics, or None if empty.
    """
    if not result.warnings:
        return None

    elem = etree.Element(f"{{{NAMESPACE}}}Diagnostics")
    for warning_text in result.warnings:
        warning_elem = etree.SubElement(
            elem,
            f"{{{NAMESPACE}}}Warning",
            severity="warning",
        )
        warning_elem.text = warning_text

    return elem


def export_uc_result_xml(result: UCResult, output_path: str) -> str:
    """Export a UCResult to XML conforming to uc_result.xsd.

    Generates a well-formed XML document with the ``urn:japan-grid:v1``
    namespace containing generator schedules, cost breakdowns, and
    diagnostics.

    Args:
        result: UCResult instance from the solver.
        output_path: File path for the output XML.

    Returns:
        The absolute path to the generated XML file.
    """
    _ensure_output_dir(output_path)

    logger.info(
        "Exporting UC result to XML: %s (status=%s, generators=%d)",
        output_path,
        result.status,
        result.num_generators,
    )

    # Root element attributes
    root_attrs = {
        "status": result.status,
        "total_cost": _format_decimal(result.total_cost),
        "solve_time_s": _format_decimal(result.solve_time_s),
    }
    if result.gap is not None:
        root_attrs["optimality_gap"] = _format_decimal(result.gap)
    root_attrs["num_generators"] = str(result.num_generators)
    if result.schedules:
        num_periods = len(result.schedules[0].commitment)
        root_attrs["num_periods"] = str(num_periods)

    # Build the XML tree
    nsmap = {None: NAMESPACE, "xsi": XSI_NAMESPACE}
    root = etree.Element(
        f"{{{NAMESPACE}}}UCResult",
        nsmap=nsmap,
        **root_attrs,
    )
    root.set(
        f"{{{XSI_NAMESPACE}}}schemaLocation",
        f"{NAMESPACE} uc_result.xsd",
    )

    # GeneratorSchedules container
    schedules_elem = etree.SubElement(
        root, f"{{{NAMESPACE}}}GeneratorSchedules"
    )
    for schedule in result.schedules:
        schedule_elem = _build_generator_schedule_element(schedule)
        schedules_elem.append(schedule_elem)

    # CostBreakdown (system-wide)
    if result.schedules:
        cost_elem = _build_cost_breakdown_element(result)
        root.append(cost_elem)

    # Diagnostics (warnings)
    diag_elem = _build_diagnostics_element(result)
    if diag_elem is not None:
        root.append(diag_elem)

    # Serialize to file
    tree = etree.ElementTree(root)
    tree.write(
        output_path,
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
    )

    logger.info(
        "UC result XML exported: %d schedules, total_cost=%.2f",
        len(result.schedules),
        result.total_cost,
    )

    return os.path.abspath(output_path)


# ------------------------------------------------------------------
# CSV Export
# ------------------------------------------------------------------


def export_uc_result_csv(result: UCResult, output_path: str) -> str:
    """Export a UCResult to CSV with one row per (generator, period).

    The CSV has the following columns:

    - ``generator_id``: Generator identifier.
    - ``period``: Time period index.
    - ``commitment``: Binary commitment status (0 or 1).
    - ``power_output_mw``: Power output in MW.
    - ``fuel_cost``: Generator total fuel cost (same for all periods).
    - ``startup_cost``: Generator total startup cost.
    - ``shutdown_cost``: Generator total shutdown cost.
    - ``no_load_cost``: Generator total no-load cost.
    - ``total_cost``: Generator total cost.

    A header comment line is included with the solve status and total cost.

    Args:
        result: UCResult instance from the solver.
        output_path: File path for the output CSV.

    Returns:
        The absolute path to the generated CSV file.
    """
    _ensure_output_dir(output_path)

    logger.info(
        "Exporting UC result to CSV: %s (status=%s, generators=%d)",
        output_path,
        result.status,
        result.num_generators,
    )

    fieldnames = [
        "generator_id",
        "period",
        "commitment",
        "power_output_mw",
        "fuel_cost",
        "startup_cost",
        "shutdown_cost",
        "no_load_cost",
        "total_cost",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        # Header comment with summary info
        f.write(
            f"# UC Result: status={result.status},"
            f" total_cost={_format_decimal(result.total_cost)},"
            f" solve_time_s={_format_decimal(result.solve_time_s)},"
            f" num_generators={result.num_generators}\n"
        )

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for schedule in result.schedules:
            for t, (commitment, power_mw) in enumerate(
                zip(schedule.commitment, schedule.power_output_mw)
            ):
                writer.writerow(
                    {
                        "generator_id": schedule.generator_id,
                        "period": t,
                        "commitment": commitment,
                        "power_output_mw": _format_decimal(power_mw),
                        "fuel_cost": _format_decimal(schedule.fuel_cost),
                        "startup_cost": _format_decimal(schedule.startup_cost),
                        "shutdown_cost": _format_decimal(
                            schedule.shutdown_cost
                        ),
                        "no_load_cost": _format_decimal(schedule.no_load_cost),
                        "total_cost": _format_decimal(schedule.total_cost),
                    }
                )

    logger.info(
        "UC result CSV exported: %d schedules, total_cost=%.2f",
        len(result.schedules),
        result.total_cost,
    )

    return os.path.abspath(output_path)
