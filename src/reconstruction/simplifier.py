"""Simplification mode: remove isolated elements from a pandapower network.

Consumes an ``IsolationResult`` from the ``Isolator`` to remove all isolated
buses (and their connected lines, generators, loads, ext_grids) from the
network, producing a single connected component suitable for power flow
analysis.

The simplifier uses ``pandapower.drop_buses()`` to cleanly remove isolated
buses and all dependent elements.  After removal it validates that at least
one ``ext_grid`` remains in service and that the resulting network forms a
single connected component.

Usage::

    from src.reconstruction.isolator import Isolator
    from src.reconstruction.simplifier import Simplifier, SimplificationResult

    isolator = Isolator()
    iso = isolator.detect(net)
    simplifier = Simplifier()
    result = simplifier.simplify(net, iso)
    print(result.summary)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import networkx as nx
import pandapower as pp
import pandapower.topology as top

from src.reconstruction.isolator import IsolationResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SimplificationResult:
    """Results from network simplification.

    Attributes:
        buses_removed: Number of isolated buses removed.
        lines_removed: Number of isolated lines removed.
        generators_removed: Number of isolated generators removed.
        loads_removed: Number of isolated loads removed.
        ext_grids_removed: Number of isolated ext_grids removed.
        buses_remaining: Number of buses remaining in the network.
        lines_remaining: Number of lines remaining in the network.
        generators_remaining: Number of generators remaining in the network.
        component_count: Number of connected components after simplification.
        ext_grid_recovered: Whether a recovery ext_grid was created.
        warnings: Non-fatal issues encountered during simplification.
    """

    buses_removed: int = 0
    lines_removed: int = 0
    generators_removed: int = 0
    loads_removed: int = 0
    ext_grids_removed: int = 0
    buses_remaining: int = 0
    lines_remaining: int = 0
    generators_remaining: int = 0
    component_count: int = 0
    ext_grid_recovered: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, object]:
        """Return a compact summary for logging."""
        return {
            "buses_removed": self.buses_removed,
            "lines_removed": self.lines_removed,
            "generators_removed": self.generators_removed,
            "loads_removed": self.loads_removed,
            "ext_grids_removed": self.ext_grids_removed,
            "buses_remaining": self.buses_remaining,
            "lines_remaining": self.lines_remaining,
            "generators_remaining": self.generators_remaining,
            "component_count": self.component_count,
            "ext_grid_recovered": self.ext_grid_recovered,
            "warnings": len(self.warnings),
        }


class Simplifier:
    """Removes isolated elements from a pandapower network.

    Given an ``IsolationResult`` that identifies all isolated elements,
    the simplifier drops isolated buses (and their dependent elements)
    to produce a network with a single connected component.  It ensures
    at least one ``ext_grid`` remains in service after removal so that
    power flow can run.

    Args:
        ensure_ext_grid: If ``True`` (default), create a recovery
            ``ext_grid`` on the main component when all ext_grids are
            lost during simplification.
    """

    def __init__(self, ensure_ext_grid: bool = True) -> None:
        self._ensure_ext_grid = ensure_ext_grid

    def simplify(
        self,
        net: Any,
        isolation_result: IsolationResult,
    ) -> SimplificationResult:
        """Remove isolated elements from the network.

        Drops all buses identified as isolated in *isolation_result*
        together with their connected lines, generators, loads, and
        ext_grids using ``pandapower.drop_buses()``.

        After removal the method validates that:

        1. The network is non-empty (at least one bus remains).
        2. At least one ``ext_grid`` is in service.
        3. The network forms a single connected component.

        The *net* object is **modified in place**.

        Args:
            net: pandapower network to simplify (modified in place).
            isolation_result: Detection results from ``Isolator.detect()``.

        Returns:
            SimplificationResult with element counts and diagnostics.

        Raises:
            ValueError: If simplification would remove *all* buses,
                leaving an empty network.
        """
        result = SimplificationResult()

        if not isolation_result.has_isolation:
            logger.info("No isolated elements detected — nothing to simplify")
            self._populate_remaining_counts(net, result)
            self._validate_components(net, result)
            return result

        # Record pre-removal counts for the isolated elements
        isolated_buses = isolation_result.isolated_buses
        isolated_lines = isolation_result.isolated_lines
        isolated_gens = isolation_result.isolated_generators
        isolated_loads = isolation_result.isolated_loads
        isolated_ext_grids = isolation_result.isolated_ext_grids

        result.buses_removed = len(isolated_buses)
        result.lines_removed = len(isolated_lines)
        result.generators_removed = len(isolated_gens)
        result.loads_removed = len(isolated_loads)
        result.ext_grids_removed = len(isolated_ext_grids)

        # Guard: refuse to create an empty network
        total_buses = len(net.bus)
        remaining = total_buses - len(isolated_buses)
        if remaining <= 0:
            raise ValueError(
                f"Simplification would remove all {total_buses} buses, "
                f"leaving an empty network. Consider using reconnection "
                f"mode instead."
            )

        logger.info(
            "Simplifying network: removing %d isolated buses "
            "(keeping %d of %d)",
            len(isolated_buses),
            remaining,
            total_buses,
        )

        # Drop isolated buses and their dependent elements
        self._drop_isolated_buses(net, isolated_buses, result)

        # Ensure at least one ext_grid remains
        if self._ensure_ext_grid:
            self._ensure_ext_grid_exists(net, isolation_result, result)

        # Post-simplification counts
        self._populate_remaining_counts(net, result)

        # Validate single connected component
        self._validate_components(net, result)

        logger.info("Simplification complete: %s", result.summary)

        return result

    # ------------------------------------------------------------------
    # Internal: bus dropping
    # ------------------------------------------------------------------

    def _drop_isolated_buses(
        self,
        net: Any,
        isolated_buses: set,
        result: SimplificationResult,
    ) -> None:
        """Drop isolated buses and all their connected elements.

        Uses ``pandapower.drop_buses()`` which removes the specified
        buses along with all lines, generators, loads, ext_grids,
        and other elements connected to them.

        Args:
            net: pandapower network (modified in place).
            isolated_buses: Set of bus indices to drop.
            result: SimplificationResult for recording warnings.
        """
        # Filter to buses that actually exist in the network
        existing_buses = set(net.bus.index)
        buses_to_drop = isolated_buses & existing_buses
        skipped = isolated_buses - existing_buses

        if skipped:
            msg = (
                f"Skipped {len(skipped)} isolated bus indices "
                f"not found in network"
            )
            result.warnings.append(msg)
            logger.warning(msg)

        if not buses_to_drop:
            msg = "No valid bus indices to drop"
            result.warnings.append(msg)
            logger.warning(msg)
            return

        try:
            pp.drop_buses(net, buses_to_drop, drop_elements=True)
            logger.info(
                "Dropped %d buses and their connected elements",
                len(buses_to_drop),
            )
        except Exception as exc:
            msg = f"Error dropping buses: {exc}"
            result.warnings.append(msg)
            logger.error(msg)
            # Fall back to deactivation if dropping fails
            self._deactivate_isolated_elements(
                net, buses_to_drop, result,
            )

    def _deactivate_isolated_elements(
        self,
        net: Any,
        isolated_buses: set,
        result: SimplificationResult,
    ) -> None:
        """Fallback: deactivate (rather than drop) isolated elements.

        Sets ``in_service = False`` on isolated buses and all elements
        connected to them.  Used when ``pp.drop_buses()`` fails.

        Args:
            net: pandapower network (modified in place).
            isolated_buses: Set of bus indices to deactivate.
            result: SimplificationResult for recording warnings.
        """
        msg = "Falling back to deactivation instead of dropping"
        result.warnings.append(msg)
        logger.warning(msg)

        # Deactivate buses
        for bus_idx in isolated_buses:
            if bus_idx in net.bus.index:
                net.bus.at[bus_idx, "in_service"] = False

        # Deactivate connected elements
        element_tables = {
            "line": ("from_bus", "to_bus"),
            "gen": ("bus",),
            "load": ("bus",),
            "ext_grid": ("bus",),
            "sgen": ("bus",),
            "trafo": ("hv_bus", "lv_bus"),
        }

        for table_name, bus_cols in element_tables.items():
            table = getattr(net, table_name, None)
            if table is None or table.empty:
                continue

            mask = False
            for col in bus_cols:
                if col in table.columns:
                    mask = mask | table[col].isin(isolated_buses)

            if hasattr(mask, "any") and mask.any():
                table.loc[mask, "in_service"] = False
                logger.info(
                    "Deactivated %d %s(s) on isolated buses",
                    mask.sum(),
                    table_name,
                )

    # ------------------------------------------------------------------
    # Internal: ext_grid recovery
    # ------------------------------------------------------------------

    def _ensure_ext_grid_exists(
        self,
        net: Any,
        isolation_result: IsolationResult,
        result: SimplificationResult,
    ) -> None:
        """Ensure the network has at least one in-service ext_grid.

        After dropping isolated elements some or all ext_grids may have
        been removed.  This method checks and, if needed, creates a
        recovery ext_grid on the highest-voltage bus in the main
        component.

        Args:
            net: pandapower network (modified in place).
            isolation_result: Original isolation result for main component
                bus information.
            result: SimplificationResult for recording warnings.
        """
        # Check if any in-service ext_grid remains
        if not net.ext_grid.empty:
            in_service_mask = net.ext_grid["in_service"]
            if in_service_mask.any():
                return

        # No ext_grid in service — create a recovery one
        if net.bus.empty:
            msg = "Cannot create recovery ext_grid: no buses remain"
            result.warnings.append(msg)
            logger.error(msg)
            return

        # Pick the highest-voltage in-service bus for the slack
        in_service_buses = net.bus[net.bus["in_service"]] if "in_service" in net.bus.columns else net.bus
        if in_service_buses.empty:
            msg = "Cannot create recovery ext_grid: no in-service buses"
            result.warnings.append(msg)
            logger.error(msg)
            return

        slack_bus = in_service_buses["vn_kv"].idxmax()

        pp.create_ext_grid(
            net,
            bus=slack_bus,
            vm_pu=1.0,
            name="slack_recovery",
        )

        result.ext_grid_recovered = True
        msg = (
            f"Created recovery ext_grid on bus {slack_bus} "
            f"({net.bus.at[slack_bus, 'vn_kv']:.0f} kV)"
        )
        result.warnings.append(msg)
        logger.warning(msg)

    # ------------------------------------------------------------------
    # Internal: post-simplification validation
    # ------------------------------------------------------------------

    def _validate_components(
        self,
        net: Any,
        result: SimplificationResult,
    ) -> None:
        """Validate that the simplified network forms a single component.

        Uses ``pandapower.topology.create_nxgraph()`` and
        ``networkx.connected_components()`` to count components.

        If more than one component remains, a warning is recorded but
        no additional modification is performed.

        Args:
            net: pandapower network (post-simplification).
            result: SimplificationResult to update with component count.
        """
        if net.bus.empty:
            result.component_count = 0
            return

        try:
            mg = top.create_nxgraph(net, respect_switches=False)
            components = list(nx.connected_components(mg))
            result.component_count = len(components)

            if len(components) == 0:
                msg = "Post-simplification network has no connected components"
                result.warnings.append(msg)
                logger.warning(msg)
            elif len(components) == 1:
                logger.info(
                    "Post-simplification network has 1 connected component "
                    "(%d buses)",
                    len(components[0]),
                )
            else:
                sizes = sorted(
                    [len(c) for c in components], reverse=True,
                )
                msg = (
                    f"Post-simplification network still has "
                    f"{len(components)} components (sizes: {sizes[:5]}"
                    f"{'...' if len(sizes) > 5 else ''})"
                )
                result.warnings.append(msg)
                logger.warning(msg)

        except Exception as exc:
            msg = f"Post-simplification topology check failed: {exc}"
            result.warnings.append(msg)
            logger.warning(msg)

    # ------------------------------------------------------------------
    # Internal: remaining element counts
    # ------------------------------------------------------------------

    def _populate_remaining_counts(
        self,
        net: Any,
        result: SimplificationResult,
    ) -> None:
        """Record remaining element counts after simplification.

        Counts only in-service elements where the ``in_service`` column
        is available.

        Args:
            net: pandapower network (post-simplification).
            result: SimplificationResult to update with counts.
        """
        result.buses_remaining = self._count_in_service(net.bus)
        result.lines_remaining = self._count_in_service(net.line)
        result.generators_remaining = self._count_in_service(net.gen)

    @staticmethod
    def _count_in_service(table: Any) -> int:
        """Count in-service rows in a pandapower element table.

        Args:
            table: A pandapower element DataFrame (e.g. ``net.bus``).

        Returns:
            Number of in-service rows, or total rows if the
            ``in_service`` column is absent.
        """
        if table.empty:
            return 0
        if "in_service" in table.columns:
            return int(table["in_service"].sum())
        return len(table)
