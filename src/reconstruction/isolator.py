"""Comprehensive isolation detection for pandapower networks.

Analyses network topology to identify isolated buses, lines, generators,
and other elements that are not part of the main connected component.
Uses pandapower's topology utilities and networkx connected component
analysis.

The Isolator class does **not** modify the network — it only analyses
and reports.  Downstream modules (Simplifier, Reconnector) consume
the IsolationResult to apply corrective actions.

Usage::

    from src.reconstruction.isolator import Isolator, IsolationResult

    isolator = Isolator()
    result = isolator.detect(net)
    print(result.summary)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import networkx as nx
import pandapower.topology as top

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class IsolationResult:
    """Results from isolation detection on a pandapower network.

    All bus indices refer to the pandapower bus DataFrame index space.

    Attributes:
        isolated_buses: Set of bus indices not in the main connected
            component.
        isolated_lines: Set of line indices where at least one endpoint
            is an isolated bus or the line is entirely within a
            non-main component.
        isolated_generators: Set of generator indices connected to
            isolated buses.
        isolated_loads: Set of load indices connected to isolated buses.
        isolated_ext_grids: Set of ext_grid indices connected to
            isolated buses.
        main_component_buses: Set of bus indices forming the largest
            connected component (the "main" network).
        component_count: Total number of connected components found.
        component_sizes: Sizes of all components sorted descending.
        warnings: Non-fatal issues encountered during detection.
    """

    isolated_buses: Set[int] = field(default_factory=set)
    isolated_lines: Set[int] = field(default_factory=set)
    isolated_generators: Set[int] = field(default_factory=set)
    isolated_loads: Set[int] = field(default_factory=set)
    isolated_ext_grids: Set[int] = field(default_factory=set)
    main_component_buses: Set[int] = field(default_factory=set)
    component_count: int = 0
    component_sizes: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_isolation(self) -> bool:
        """Return True if any isolated elements were detected."""
        return bool(
            self.isolated_buses
            or self.isolated_lines
            or self.isolated_generators
        )

    @property
    def summary(self) -> Dict[str, object]:
        """Return a compact summary for logging."""
        return {
            "component_count": self.component_count,
            "main_component_size": len(self.main_component_buses),
            "isolated_buses": len(self.isolated_buses),
            "isolated_lines": len(self.isolated_lines),
            "isolated_generators": len(self.isolated_generators),
            "isolated_loads": len(self.isolated_loads),
            "isolated_ext_grids": len(self.isolated_ext_grids),
            "has_isolation": self.has_isolation,
            "warnings": len(self.warnings),
        }


class Isolator:
    """Detects isolated elements in a pandapower network.

    Uses ``pandapower.topology.create_nxgraph()`` and
    ``networkx.connected_components()`` to identify connected components,
    then classifies buses, lines, generators, loads, and ext_grids as
    isolated if they are not part of the largest connected component.

    Args:
        min_component_size: Minimum bus count for a component to be
            considered viable.  Components smaller than this threshold
            are treated as isolated even if they have multiple buses.
            Defaults to 2.
        respect_switches: Whether to respect switch states when building
            the topology graph.  Defaults to ``False`` (treat all
            switches as closed).
    """

    def __init__(
        self,
        min_component_size: int = 2,
        respect_switches: bool = False,
    ) -> None:
        if min_component_size < 1:
            raise ValueError(
                f"min_component_size must be >= 1, got {min_component_size}"
            )
        self._min_component_size = min_component_size
        self._respect_switches = respect_switches

    def detect(self, net: Any) -> IsolationResult:
        """Detect all isolated elements in a pandapower network.

        Performs topology analysis to identify isolated buses and then
        determines which lines, generators, loads, and ext_grids are
        connected to those isolated buses.

        The network is **not modified** — only analysed.

        Args:
            net: A pandapower network to analyse.

        Returns:
            IsolationResult with sets of isolated element indices and
            topology statistics.
        """
        result = IsolationResult()

        # Validate that the network has buses
        if net.bus.empty:
            result.warnings.append("Network has no buses")
            logger.warning("Network has no buses — nothing to analyse")
            return result

        # Step 1: Build topology graph and find connected components
        self._find_components(net, result)

        # Step 2: Identify isolated elements by association
        if result.isolated_buses:
            self._find_isolated_lines(net, result)
            self._find_isolated_generators(net, result)
            self._find_isolated_loads(net, result)
            self._find_isolated_ext_grids(net, result)

        logger.info("Isolation detection complete: %s", result.summary)

        return result

    # ------------------------------------------------------------------
    # Internal: connected component analysis
    # ------------------------------------------------------------------

    def _find_components(self, net: Any, result: IsolationResult) -> None:
        """Build topology graph and identify connected components.

        Uses pandapower's ``create_nxgraph()`` which considers in-service
        buses and lines.  Only buses that are in service and connected
        via at least one in-service line appear as nodes in the graph.

        Buses not present in the graph at all (completely disconnected)
        are also treated as isolated.

        Args:
            net: pandapower network.
            result: IsolationResult to populate with component data.
        """
        try:
            mg = top.create_nxgraph(
                net,
                respect_switches=self._respect_switches,
            )
        except Exception as exc:
            msg = f"Failed to create topology graph: {exc}"
            result.warnings.append(msg)
            logger.warning(msg)
            return

        # Find all connected components
        components = list(nx.connected_components(mg))
        result.component_count = len(components)
        result.component_sizes = sorted(
            [len(c) for c in components], reverse=True,
        )

        if not components:
            # No components means no connected buses at all
            all_in_service = set(
                net.bus.index[net.bus["in_service"]]
                if "in_service" in net.bus.columns
                else net.bus.index
            )
            result.isolated_buses = all_in_service
            result.warnings.append(
                "No connected components found — all buses are isolated"
            )
            logger.warning("No connected components found")
            return

        # Identify the main (largest) component
        largest = max(components, key=len)
        result.main_component_buses = set(largest)

        # Check for small-component edge case: if the largest component
        # itself is smaller than min_component_size, treat all as isolated
        if len(largest) < self._min_component_size:
            msg = (
                f"Largest component has only {len(largest)} bus(es), "
                f"below min_component_size={self._min_component_size}"
            )
            result.warnings.append(msg)
            logger.warning(msg)

        # Collect buses from non-main components
        for comp in components:
            if comp is not largest:
                result.isolated_buses.update(comp)

        # Also check for in-service buses not present in any component
        # (completely disconnected buses with no lines at all)
        graph_buses = set(mg.nodes())
        all_bus_indices = set(
            net.bus.index[net.bus["in_service"]]
            if "in_service" in net.bus.columns
            else net.bus.index
        )
        disconnected = all_bus_indices - graph_buses
        if disconnected:
            result.isolated_buses.update(disconnected)
            msg = (
                f"Found {len(disconnected)} bus(es) with no line "
                f"connections (completely disconnected)"
            )
            result.warnings.append(msg)
            logger.warning(msg)

        if result.isolated_buses:
            sizes_str = str(result.component_sizes[:5])
            if len(result.component_sizes) > 5:
                sizes_str = sizes_str[:-1] + ", ...]"
            logger.info(
                "Network has %d connected component(s) "
                "(sizes: %s), %d isolated bus(es)",
                result.component_count,
                sizes_str,
                len(result.isolated_buses),
            )

    # ------------------------------------------------------------------
    # Internal: isolated element identification
    # ------------------------------------------------------------------

    def _find_isolated_lines(
        self, net: Any, result: IsolationResult,
    ) -> None:
        """Identify lines connected to isolated buses.

        A line is considered isolated if **either** its ``from_bus``
        or ``to_bus`` is in the isolated bus set.

        Args:
            net: pandapower network.
            result: IsolationResult with populated ``isolated_buses``.
        """
        if net.line.empty:
            return

        isolated = result.isolated_buses

        for idx in net.line.index:
            from_bus = net.line.at[idx, "from_bus"]
            to_bus = net.line.at[idx, "to_bus"]
            if from_bus in isolated or to_bus in isolated:
                result.isolated_lines.add(idx)

    def _find_isolated_generators(
        self, net: Any, result: IsolationResult,
    ) -> None:
        """Identify generators connected to isolated buses.

        A generator is isolated if its ``bus`` is in the isolated bus set.

        Args:
            net: pandapower network.
            result: IsolationResult with populated ``isolated_buses``.
        """
        if net.gen.empty:
            return

        isolated = result.isolated_buses

        for idx in net.gen.index:
            if net.gen.at[idx, "bus"] in isolated:
                result.isolated_generators.add(idx)

    def _find_isolated_loads(
        self, net: Any, result: IsolationResult,
    ) -> None:
        """Identify loads connected to isolated buses.

        A load is isolated if its ``bus`` is in the isolated bus set.

        Args:
            net: pandapower network.
            result: IsolationResult with populated ``isolated_buses``.
        """
        if net.load.empty:
            return

        isolated = result.isolated_buses

        for idx in net.load.index:
            if net.load.at[idx, "bus"] in isolated:
                result.isolated_loads.add(idx)

    def _find_isolated_ext_grids(
        self, net: Any, result: IsolationResult,
    ) -> None:
        """Identify external grids connected to isolated buses.

        An ext_grid is isolated if its ``bus`` is in the isolated bus set.

        Args:
            net: pandapower network.
            result: IsolationResult with populated ``isolated_buses``.
        """
        if net.ext_grid.empty:
            return

        isolated = result.isolated_buses

        for idx in net.ext_grid.index:
            if net.ext_grid.at[idx, "bus"] in isolated:
                result.isolated_ext_grids.add(idx)

    # ------------------------------------------------------------------
    # Convenience: unsupplied bus detection
    # ------------------------------------------------------------------

    @staticmethod
    def find_unsupplied_buses(net: Any) -> Set[int]:
        """Find buses that have no path to any ext_grid (slack bus).

        Uses pandapower's built-in ``unsupplied_buses()`` utility.
        These buses cannot receive power and will cause convergence
        failures if not handled.

        Args:
            net: pandapower network.

        Returns:
            Set of bus indices that are unsupplied.
        """
        try:
            unsupplied = top.unsupplied_buses(
                net, respect_switches=False,
            )
            return set(unsupplied)
        except Exception as exc:
            logger.warning("Failed to detect unsupplied buses: %s", exc)
            return set()
