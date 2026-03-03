"""DC and AC power flow computation for pandapower networks.

Runs DC power flow (always converges) as the primary method and
optionally attempts AC power flow.  Includes topology validation
to detect isolated buses before computation.

Usage::

    from src.powerflow.powerflow_runner import run_powerflow

    result = run_powerflow(net, mode="dc")
    print(result.total_loss_mw)
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd
import pandapower as pp
import pandapower.topology as top

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PowerFlowResult:
    """Results from a power flow computation.

    Attributes:
        converged: Whether the power flow converged.
        mode: Computation mode (``"dc"`` or ``"ac"``).
        res_bus: Bus results DataFrame (vm_pu, va_degree, p_mw, q_mvar).
        res_line: Line results DataFrame (loading_percent, p_from_mw, etc.).
        res_gen: Generator results DataFrame (p_mw, q_mvar).
        res_ext_grid: External grid results DataFrame.
        total_loss_mw: Total active power losses (MW).
        max_line_loading_pct: Maximum line loading percentage.
        warnings: Non-fatal issues encountered.
    """

    converged: bool = False
    mode: str = "dc"
    res_bus: Optional[pd.DataFrame] = None
    res_line: Optional[pd.DataFrame] = None
    res_gen: Optional[pd.DataFrame] = None
    res_ext_grid: Optional[pd.DataFrame] = None
    total_loss_mw: float = 0.0
    max_line_loading_pct: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def summary(self) -> dict:
        """Return a compact summary for logging."""
        return {
            "converged": self.converged,
            "mode": self.mode,
            "total_loss_mw": round(self.total_loss_mw, 2),
            "max_line_loading_pct": round(self.max_line_loading_pct, 2),
            "warnings": len(self.warnings),
        }


def run_powerflow(
    net: Any,
    mode: str = "dc",
    ac_fallback: bool = False,
) -> PowerFlowResult:
    """Run power flow analysis on a pandapower network.

    Args:
        net: pandapower network (modified in place with results).
        mode: ``"dc"`` for DC power flow, ``"ac"`` for AC power flow.
        ac_fallback: If ``True`` and *mode* is ``"dc"``, also attempt
            AC power flow (non-fatal on failure).

    Returns:
        PowerFlowResult with bus/line/gen DataFrames and summary metrics.
    """
    result = PowerFlowResult(mode=mode)

    # Pre-flight topology check
    _check_topology(net, result)

    if mode == "dc":
        _run_dc(net, result)
    elif mode == "ac":
        _run_ac(net, result)
    else:
        logger.error("Unknown power flow mode: '%s'", mode)
        result.warnings.append(f"Unknown mode '{mode}'; defaulting to DC")
        _run_dc(net, result)

    # Optional AC attempt after DC
    if ac_fallback and mode == "dc" and result.converged:
        logger.info("Attempting AC power flow (fallback)...")
        ac_result = PowerFlowResult(mode="ac")
        _run_ac(net, ac_result)
        if ac_result.converged:
            logger.info("AC power flow also converged")
        else:
            logger.info("AC power flow did not converge (non-fatal)")

    return result


def _check_topology(net: Any, result: PowerFlowResult) -> None:
    """Check network topology for isolated components."""
    try:
        # Get connected components using pandapower topology
        mg = top.create_nxgraph(net, respect_switches=False)
        import networkx as nx
        components = list(nx.connected_components(mg))

        if len(components) > 1:
            sizes = sorted([len(c) for c in components], reverse=True)
            msg = (
                f"Network has {len(components)} connected components "
                f"(sizes: {sizes[:5]}{'...' if len(sizes) > 5 else ''})"
            )
            result.warnings.append(msg)
            logger.warning(msg)

            # Deactivate isolated buses (not in the largest component)
            largest = max(components, key=len)
            isolated_buses = set()
            for comp in components:
                if comp != largest:
                    isolated_buses.update(comp)

            if isolated_buses:
                for bus_idx in isolated_buses:
                    if bus_idx in net.bus.index:
                        net.bus.at[bus_idx, "in_service"] = False

                # Deactivate loads/gens/lines connected to isolated buses
                for table_name in ("load", "gen", "line"):
                    table = getattr(net, table_name, None)
                    if table is None or table.empty:
                        continue
                    if table_name == "line":
                        mask = (
                            table["from_bus"].isin(isolated_buses)
                            | table["to_bus"].isin(isolated_buses)
                        )
                    else:
                        mask = table["bus"].isin(isolated_buses)
                    table.loc[mask, "in_service"] = False

                # Deactivate ext_grids on isolated buses
                if not net.ext_grid.empty:
                    mask = net.ext_grid["bus"].isin(isolated_buses)
                    net.ext_grid.loc[mask, "in_service"] = False

                msg = (
                    f"Deactivated {len(isolated_buses)} isolated buses "
                    f"and their connected elements"
                )
                result.warnings.append(msg)
                logger.warning(msg)

                # Ensure at least one ext_grid is in service
                if net.ext_grid["in_service"].sum() == 0 and len(net.ext_grid) > 0:
                    # Re-enable the first ext_grid on a bus in the largest component
                    for i, row in net.ext_grid.iterrows():
                        if row["bus"] in largest:
                            net.ext_grid.at[i, "in_service"] = True
                            logger.info(
                                "Re-enabled ext_grid %d on bus %d (largest component)",
                                i, row["bus"],
                            )
                            break
                    else:
                        # Create a new ext_grid on a bus in the largest component
                        bus_idx = next(iter(largest))
                        pp.create_ext_grid(net, bus=bus_idx, vm_pu=1.0, name="slack_recovery")
                        msg = "Created recovery ext_grid on bus in largest component"
                        result.warnings.append(msg)
                        logger.warning(msg)

    except Exception as exc:
        msg = f"Topology check failed: {exc}"
        result.warnings.append(msg)
        logger.warning(msg)


def _run_dc(net: Any, result: PowerFlowResult) -> None:
    """Execute DC power flow."""
    try:
        pp.rundcpp(net)
        result.converged = True
        result.mode = "dc"
        _extract_results(net, result)
        logger.info(
            "DC power flow converged: loss=%.1f MW, max_loading=%.1f%%",
            result.total_loss_mw,
            result.max_line_loading_pct,
        )
    except Exception as exc:
        result.converged = False
        msg = f"DC power flow failed: {exc}"
        result.warnings.append(msg)
        logger.error(msg)


def _run_ac(net: Any, result: PowerFlowResult) -> None:
    """Execute AC power flow."""
    try:
        pp.runpp(net, numba=False)
        result.converged = True
        result.mode = "ac"
        _extract_results(net, result)
        logger.info(
            "AC power flow converged: loss=%.1f MW, max_loading=%.1f%%",
            result.total_loss_mw,
            result.max_line_loading_pct,
        )
    except Exception as exc:
        result.converged = False
        msg = f"AC power flow did not converge: {exc}"
        result.warnings.append(msg)
        logger.warning(msg)


def _extract_results(net: Any, result: PowerFlowResult) -> None:
    """Extract result DataFrames and compute summary metrics."""
    result.res_bus = net.res_bus.copy() if hasattr(net, "res_bus") else None
    result.res_line = net.res_line.copy() if hasattr(net, "res_line") else None
    result.res_gen = net.res_gen.copy() if hasattr(net, "res_gen") else None
    result.res_ext_grid = (
        net.res_ext_grid.copy() if hasattr(net, "res_ext_grid") else None
    )

    # Total loss
    if result.res_line is not None and "pl_mw" in result.res_line.columns:
        result.total_loss_mw = result.res_line["pl_mw"].sum()

    # Max line loading
    if result.res_line is not None and "loading_percent" in result.res_line.columns:
        in_service = net.line["in_service"] if "in_service" in net.line.columns else True
        active_loading = result.res_line.loc[in_service, "loading_percent"]
        if len(active_loading) > 0:
            result.max_line_loading_pct = active_loading.max()
