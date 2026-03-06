"""Reconnection mode: generate synthetic connections for isolated elements.

Consumes an ``IsolationResult`` from the ``Isolator`` to create synthetic
transmission lines that reconnect isolated buses to the main connected
component.  After reconnection the module validates the resulting Ybus
admittance matrix to ensure it is correctly sized and non-singular.

Synthetic lines are created using reference electrical parameters from
``config/line_types.yaml`` via :func:`get_line_parameters_safe`, with
minimum reactance enforcement to prevent Ybus singularity.

The nearest main-component bus is found via geodata (Haversine distance).
When geodata is unavailable, the first main-component bus at a matching
voltage level is used as a fallback.

Usage::

    from src.reconstruction.isolator import Isolator
    from src.reconstruction.reconnector import Reconnector, ReconnectionResult
    from src.reconstruction.config import ReconstructionConfig

    isolator = Isolator()
    iso = isolator.detect(net)
    config = ReconstructionConfig(mode="reconnect")
    reconnector = Reconnector()
    result = reconnector.reconnect(net, iso, config)
    print(result.summary)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandapower as pp
from scipy import sparse

from src.converter.line_parameters import get_line_parameters_safe
from src.reconstruction.config import ReconstructionConfig
from src.reconstruction.isolator import IsolationResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# Earth radius in kilometres for Haversine distance
_EARTH_RADIUS_KM = 6371.0

# Default synthetic line length when geodata is unavailable (km)
_DEFAULT_SYNTHETIC_LENGTH_KM = 10.0

# Minimum synthetic line length to avoid degenerate zero-length branches (km)
_MIN_LINE_LENGTH_KM = 0.1


@dataclass
class ReconnectionResult:
    """Results from reconnecting isolated elements in a pandapower network.

    Attributes:
        lines_created: Number of synthetic reconnection lines created.
        buses_reconnected: Number of isolated buses successfully reconnected.
        buses_unreachable: Number of isolated buses that could not be
            reconnected (e.g. beyond max distance or no main component).
        ybus_shape: Shape of the Ybus matrix after reconnection, or
            ``None`` if Ybus extraction failed.
        ybus_nonsingular: Whether the Ybus matrix is non-singular
            (i.e. has a non-zero determinant / full rank).
        synthetic_line_map: Mapping from synthetic line name to
            pandapower line index.
        warnings: Non-fatal issues encountered during reconnection.
    """

    lines_created: int = 0
    buses_reconnected: int = 0
    buses_unreachable: int = 0
    ybus_shape: Optional[Tuple[int, int]] = None
    ybus_nonsingular: bool = False
    synthetic_line_map: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, object]:
        """Return a compact summary for logging."""
        return {
            "lines_created": self.lines_created,
            "buses_reconnected": self.buses_reconnected,
            "buses_unreachable": self.buses_unreachable,
            "ybus_shape": self.ybus_shape,
            "ybus_nonsingular": self.ybus_nonsingular,
            "warnings": len(self.warnings),
        }


class Reconnector:
    """Reconnects isolated buses to the main network component.

    For each isolated bus, the reconnector finds the nearest
    main-component bus (via Haversine distance on geodata) and creates
    a synthetic transmission line between them.  Line electrical
    parameters are obtained from the Japanese reference table via
    :func:`get_line_parameters_safe`.

    After all reconnections, the Ybus admittance matrix is extracted
    and validated for correct dimensions and non-singularity.

    The *net* object is **modified in place**.
    """

    def reconnect(
        self,
        net: Any,
        isolation_result: IsolationResult,
        config: ReconstructionConfig,
    ) -> ReconnectionResult:
        """Reconnect isolated buses to the main connected component.

        For each isolated bus:

        1. Determine the bus voltage (or use the configured default).
        2. Find the nearest main-component bus via geodata distance.
        3. Create a synthetic transmission line with reference parameters.
        4. Enforce minimum reactance to prevent Ybus singularity.

        After all lines are created, the Ybus matrix is extracted
        and validated.

        Args:
            net: pandapower network to reconnect (modified in place).
            isolation_result: Detection results from ``Isolator.detect()``.
            config: Reconstruction configuration with thresholds and
                defaults.

        Returns:
            ReconnectionResult with counts, Ybus validation status,
            and any warnings.
        """
        result = ReconnectionResult()

        if not isolation_result.has_isolation:
            logger.info(
                "No isolated elements detected — nothing to reconnect"
            )
            self._validate_ybus(net, result)
            return result

        isolated_buses = isolation_result.isolated_buses
        main_buses = isolation_result.main_component_buses

        # Edge case: no main component (all buses isolated)
        if not main_buses:
            self._handle_all_isolated(net, isolated_buses, config, result)
            return result

        logger.info(
            "Reconnecting %d isolated bus(es) to main component "
            "(%d buses)",
            len(isolated_buses),
            len(main_buses),
        )

        # Ensure ext_grid exists before reconnection
        self._ensure_ext_grid(net, main_buses, result)

        # Resolve network frequency
        f_hz = self._resolve_frequency(net)

        # Build geodata lookup for main-component buses
        main_geodata = self._get_bus_geodata(net, main_buses)

        # Reconnect each isolated bus
        for bus_idx in sorted(isolated_buses):
            self._reconnect_bus(
                net=net,
                bus_idx=bus_idx,
                main_buses=main_buses,
                main_geodata=main_geodata,
                f_hz=f_hz,
                config=config,
                result=result,
            )

        # Validate Ybus after reconnection
        self._validate_ybus(net, result)

        logger.info("Reconnection complete: %s", result.summary)

        return result

    # ------------------------------------------------------------------
    # Internal: per-bus reconnection
    # ------------------------------------------------------------------

    def _reconnect_bus(
        self,
        net: Any,
        bus_idx: int,
        main_buses: Set[int],
        main_geodata: Dict[int, Tuple[float, float]],
        f_hz: float,
        config: ReconstructionConfig,
        result: ReconnectionResult,
    ) -> None:
        """Reconnect a single isolated bus to the main component.

        Finds the nearest main-component bus and creates a synthetic
        line between them.

        Args:
            net: pandapower network (modified in place).
            bus_idx: Index of the isolated bus.
            main_buses: Set of bus indices in the main component.
            main_geodata: Geodata mapping for main-component buses.
            f_hz: Network frequency in Hz.
            config: Reconstruction configuration.
            result: ReconnectionResult to update.
        """
        # Verify the isolated bus still exists
        if bus_idx not in net.bus.index:
            msg = f"Isolated bus {bus_idx} not found in network — skipping"
            result.warnings.append(msg)
            logger.warning(msg)
            result.buses_unreachable += 1
            return

        # Determine voltage for the synthetic line
        voltage_kv = self._resolve_bus_voltage(
            net, bus_idx, config.default_voltage_kv,
        )

        # Find the nearest main-component bus
        target_bus, distance_km = self._find_nearest_main_bus(
            net=net,
            bus_idx=bus_idx,
            main_buses=main_buses,
            main_geodata=main_geodata,
            voltage_kv=voltage_kv,
            max_distance_km=config.max_reconnection_distance_km,
        )

        if target_bus is None:
            msg = (
                f"Bus {bus_idx}: no main-component bus found within "
                f"{config.max_reconnection_distance_km:.0f} km — "
                f"cannot reconnect"
            )
            result.warnings.append(msg)
            logger.warning(msg)
            result.buses_unreachable += 1
            return

        # Determine synthetic line length
        line_length_km = max(distance_km, _MIN_LINE_LENGTH_KM)

        # Get electrical parameters for the synthetic line
        r_ohm, x_ohm, c_nf, max_i = self._get_synthetic_line_params(
            voltage_kv=voltage_kv,
            f_hz=f_hz,
            min_reactance=config.min_reactance_ohm_per_km,
            result=result,
        )

        # Create the synthetic reconnection line
        line_name = f"recon_line_{bus_idx}_{target_bus}"

        line_idx = pp.create_line_from_parameters(
            net,
            from_bus=bus_idx,
            to_bus=target_bus,
            length_km=line_length_km,
            r_ohm_per_km=r_ohm,
            x_ohm_per_km=x_ohm,
            c_nf_per_km=c_nf,
            max_i_ka=max_i,
            name=line_name,
        )

        result.synthetic_line_map[line_name] = line_idx
        result.lines_created += 1
        result.buses_reconnected += 1

        logger.debug(
            "Reconnected bus %d → bus %d via '%s' "
            "(%.1f km, %.0f kV, line_idx=%d)",
            bus_idx,
            target_bus,
            line_name,
            line_length_km,
            voltage_kv,
            line_idx,
        )

    # ------------------------------------------------------------------
    # Internal: nearest bus search
    # ------------------------------------------------------------------

    def _find_nearest_main_bus(
        self,
        net: Any,
        bus_idx: int,
        main_buses: Set[int],
        main_geodata: Dict[int, Tuple[float, float]],
        voltage_kv: float,
        max_distance_km: float,
    ) -> Tuple[Optional[int], float]:
        """Find the nearest main-component bus to an isolated bus.

        Uses Haversine distance on geodata when available.  Falls back
        to voltage-matching when geodata is unavailable.

        Args:
            net: pandapower network.
            bus_idx: Index of the isolated bus.
            main_buses: Set of main-component bus indices.
            main_geodata: Geodata mapping for main-component buses.
            voltage_kv: Voltage of the isolated bus (for matching).
            max_distance_km: Maximum search radius in km.

        Returns:
            Tuple of (target_bus_idx, distance_km).  Returns
            ``(None, inf)`` if no suitable bus is found.
        """
        iso_geo = self._get_single_bus_geodata(net, bus_idx)

        # Strategy 1: Geodata-based nearest bus
        if iso_geo is not None and main_geodata:
            best_bus = None
            best_dist = float("inf")

            for m_bus, m_geo in main_geodata.items():
                dist = _haversine_km(iso_geo, m_geo)
                if dist < best_dist:
                    best_dist = dist
                    best_bus = m_bus

            if best_bus is not None and best_dist <= max_distance_km:
                return best_bus, best_dist

            # If beyond max distance, try voltage-matched bus within limit
            if best_bus is not None and best_dist > max_distance_km:
                voltage_matched = self._find_voltage_matched_bus(
                    net, main_geodata, iso_geo, voltage_kv,
                    max_distance_km,
                )
                if voltage_matched is not None:
                    return voltage_matched

        # Strategy 2: Fallback — voltage-matched or first available
        return self._find_fallback_main_bus(
            net, main_buses, voltage_kv,
        )

    def _find_voltage_matched_bus(
        self,
        net: Any,
        main_geodata: Dict[int, Tuple[float, float]],
        iso_geo: Tuple[float, float],
        voltage_kv: float,
        max_distance_km: float,
    ) -> Optional[Tuple[int, float]]:
        """Find the nearest voltage-matched main bus within distance limit.

        Args:
            net: pandapower network.
            main_geodata: Geodata for main-component buses.
            iso_geo: Geodata of the isolated bus.
            voltage_kv: Desired voltage for matching.
            max_distance_km: Maximum search radius.

        Returns:
            Tuple of (bus_idx, distance_km) or None.
        """
        best_bus = None
        best_dist = float("inf")

        for m_bus, m_geo in main_geodata.items():
            if m_bus not in net.bus.index:
                continue
            m_voltage = net.bus.at[m_bus, "vn_kv"]
            if abs(m_voltage - voltage_kv) < 1.0:  # Same voltage class
                dist = _haversine_km(iso_geo, m_geo)
                if dist < best_dist and dist <= max_distance_km:
                    best_dist = dist
                    best_bus = m_bus

        if best_bus is not None:
            return best_bus, best_dist
        return None

    def _find_fallback_main_bus(
        self,
        net: Any,
        main_buses: Set[int],
        voltage_kv: float,
    ) -> Tuple[Optional[int], float]:
        """Find a fallback main-component bus without geodata.

        Tries to match by voltage first, then falls back to the
        highest-voltage main bus.

        Args:
            net: pandapower network.
            main_buses: Set of main-component bus indices.
            voltage_kv: Desired voltage for matching.

        Returns:
            Tuple of (bus_idx, default_distance).  Returns
            ``(None, inf)`` if main_buses is empty.
        """
        if not main_buses:
            return None, float("inf")

        # Filter to buses that exist in the network
        valid_main = [b for b in main_buses if b in net.bus.index]
        if not valid_main:
            return None, float("inf")

        # Try voltage-matched first
        for bus_idx in valid_main:
            bus_voltage = net.bus.at[bus_idx, "vn_kv"]
            if abs(bus_voltage - voltage_kv) < 1.0:
                return bus_idx, _DEFAULT_SYNTHETIC_LENGTH_KM

        # Fall back to highest-voltage bus in the main component
        best_bus = max(
            valid_main,
            key=lambda b: net.bus.at[b, "vn_kv"],
        )
        return best_bus, _DEFAULT_SYNTHETIC_LENGTH_KM

    # ------------------------------------------------------------------
    # Internal: geodata utilities
    # ------------------------------------------------------------------

    def _get_bus_geodata(
        self,
        net: Any,
        bus_indices: Set[int],
    ) -> Dict[int, Tuple[float, float]]:
        """Extract geodata (longitude, latitude) for a set of buses.

        Args:
            net: pandapower network.
            bus_indices: Bus indices to extract geodata for.

        Returns:
            Dictionary mapping bus index to (longitude, latitude).
            Only buses with valid geodata are included.
        """
        geodata: Dict[int, Tuple[float, float]] = {}

        if not hasattr(net, "bus_geodata") or net.bus_geodata.empty:
            return geodata

        for bus_idx in bus_indices:
            geo = self._get_single_bus_geodata(net, bus_idx)
            if geo is not None:
                geodata[bus_idx] = geo

        return geodata

    @staticmethod
    def _get_single_bus_geodata(
        net: Any,
        bus_idx: int,
    ) -> Optional[Tuple[float, float]]:
        """Get geodata for a single bus.

        Args:
            net: pandapower network.
            bus_idx: Bus index.

        Returns:
            Tuple of (longitude, latitude) or None if unavailable.
        """
        if not hasattr(net, "bus_geodata") or net.bus_geodata.empty:
            return None

        if bus_idx not in net.bus_geodata.index:
            return None

        x = net.bus_geodata.at[bus_idx, "x"]
        y = net.bus_geodata.at[bus_idx, "y"]

        # Validate coordinates
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        if x == 0.0 and y == 0.0:
            return None

        return (float(x), float(y))

    # ------------------------------------------------------------------
    # Internal: voltage and frequency resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_bus_voltage(
        net: Any,
        bus_idx: int,
        default_kv: float,
    ) -> float:
        """Resolve voltage for a bus, using default if zero/missing.

        Args:
            net: pandapower network.
            bus_idx: Bus index.
            default_kv: Fallback voltage in kV.

        Returns:
            Bus voltage in kV.
        """
        voltage = net.bus.at[bus_idx, "vn_kv"]
        if voltage > 0:
            return float(voltage)
        return default_kv

    @staticmethod
    def _resolve_frequency(net: Any) -> float:
        """Resolve the network frequency.

        Args:
            net: pandapower network.

        Returns:
            Network frequency in Hz (defaults to 50 if unset).
        """
        f_hz = getattr(net, "f_hz", 50)
        if f_hz not in (50, 60):
            return 50
        return float(f_hz)

    # ------------------------------------------------------------------
    # Internal: synthetic line parameters
    # ------------------------------------------------------------------

    def _get_synthetic_line_params(
        self,
        voltage_kv: float,
        f_hz: float,
        min_reactance: float,
        result: ReconnectionResult,
    ) -> Tuple[float, float, float, float]:
        """Get electrical parameters for a synthetic reconnection line.

        Uses :func:`get_line_parameters_safe` for voltage-class lookup
        with fallback.  Enforces minimum reactance to prevent Ybus
        singularity.

        Args:
            voltage_kv: Line voltage in kV.
            f_hz: System frequency in Hz.
            min_reactance: Minimum reactance in Ohm/km.
            result: ReconnectionResult for recording warnings.

        Returns:
            Tuple of (r_ohm_per_km, x_ohm_per_km, c_nf_per_km, max_i_ka).
        """
        params = get_line_parameters_safe(voltage_kv, f_hz)

        if params is not None:
            r_ohm = params["r_ohm_per_km"]
            x_ohm = params["x_ohm_per_km"]
            c_nf = params["c_nf_per_km"]
            max_i = params["max_i_ka"]
        else:
            # Last resort: generic defaults (same as pandapower_builder)
            msg = (
                f"No line parameters for {voltage_kv:.0f} kV; "
                f"using generic defaults for synthetic line"
            )
            result.warnings.append(msg)
            logger.warning(msg)
            r_ohm = 0.05
            x_ohm = 0.4
            c_nf = 10.0
            max_i = 1.0

        # Enforce minimum reactance
        if x_ohm < min_reactance:
            x_ohm = min_reactance
            logger.debug(
                "Reactance %.4f Ohm/km below minimum; "
                "set to %.4f Ohm/km",
                x_ohm,
                min_reactance,
            )

        return r_ohm, x_ohm, c_nf, max_i

    # ------------------------------------------------------------------
    # Internal: edge case — all buses isolated
    # ------------------------------------------------------------------

    def _handle_all_isolated(
        self,
        net: Any,
        isolated_buses: Set[int],
        config: ReconstructionConfig,
        result: ReconnectionResult,
    ) -> None:
        """Handle the edge case where all buses are isolated.

        Creates a synthetic central hub bus and connects all isolated
        buses to it via synthetic lines.

        Args:
            net: pandapower network (modified in place).
            isolated_buses: All bus indices (all isolated).
            config: Reconstruction configuration.
            result: ReconnectionResult to update.
        """
        msg = (
            "No main component found — all buses are isolated. "
            "Creating synthetic central hub for reconnection."
        )
        result.warnings.append(msg)
        logger.warning(msg)

        if not isolated_buses:
            return

        # Determine hub voltage: use the highest voltage among isolated buses
        valid_buses = [b for b in isolated_buses if b in net.bus.index]
        if not valid_buses:
            return

        hub_voltage = max(
            net.bus.at[b, "vn_kv"] for b in valid_buses
        )
        if hub_voltage <= 0:
            hub_voltage = config.default_voltage_kv

        # Create the central hub bus
        hub_bus = pp.create_bus(
            net,
            vn_kv=hub_voltage,
            name="recon_hub",
        )

        # Create ext_grid on the hub bus (slack)
        pp.create_ext_grid(
            net,
            bus=hub_bus,
            vm_pu=1.0,
            name="recon_hub_slack",
        )

        f_hz = self._resolve_frequency(net)

        # Connect each isolated bus to the hub
        for bus_idx in sorted(valid_buses):
            voltage_kv = self._resolve_bus_voltage(
                net, bus_idx, config.default_voltage_kv,
            )

            r_ohm, x_ohm, c_nf, max_i = self._get_synthetic_line_params(
                voltage_kv=voltage_kv,
                f_hz=f_hz,
                min_reactance=config.min_reactance_ohm_per_km,
                result=result,
            )

            line_name = f"recon_line_{bus_idx}_{hub_bus}"

            line_idx = pp.create_line_from_parameters(
                net,
                from_bus=bus_idx,
                to_bus=hub_bus,
                length_km=_DEFAULT_SYNTHETIC_LENGTH_KM,
                r_ohm_per_km=r_ohm,
                x_ohm_per_km=x_ohm,
                c_nf_per_km=c_nf,
                max_i_ka=max_i,
                name=line_name,
            )

            result.synthetic_line_map[line_name] = line_idx
            result.lines_created += 1
            result.buses_reconnected += 1

        logger.info(
            "Created central hub (bus %d) and %d synthetic lines",
            hub_bus,
            result.lines_created,
        )

        # Validate Ybus
        self._validate_ybus(net, result)

    # ------------------------------------------------------------------
    # Internal: ext_grid safety
    # ------------------------------------------------------------------

    def _ensure_ext_grid(
        self,
        net: Any,
        main_buses: Set[int],
        result: ReconnectionResult,
    ) -> None:
        """Ensure the network has at least one in-service ext_grid.

        If no ext_grid exists, creates one on the highest-voltage bus
        in the main component.

        Args:
            net: pandapower network (modified in place).
            main_buses: Set of main-component bus indices.
            result: ReconnectionResult for recording warnings.
        """
        if not net.ext_grid.empty:
            in_service = net.ext_grid["in_service"]
            if in_service.any():
                return

        # Create ext_grid on the highest-voltage main bus
        valid = [b for b in main_buses if b in net.bus.index]
        if not valid:
            msg = "Cannot create ext_grid: no valid main-component buses"
            result.warnings.append(msg)
            logger.error(msg)
            return

        slack_bus = max(valid, key=lambda b: net.bus.at[b, "vn_kv"])

        pp.create_ext_grid(
            net,
            bus=slack_bus,
            vm_pu=1.0,
            name="recon_slack",
        )

        msg = (
            f"Created reconnection ext_grid on bus {slack_bus} "
            f"({net.bus.at[slack_bus, 'vn_kv']:.0f} kV)"
        )
        result.warnings.append(msg)
        logger.warning(msg)

    # ------------------------------------------------------------------
    # Internal: Ybus validation
    # ------------------------------------------------------------------

    def _validate_ybus(
        self,
        net: Any,
        result: ReconnectionResult,
    ) -> None:
        """Extract and validate the Ybus admittance matrix.

        Runs ``pp.runpp()`` to populate ``net._ppc`` and then checks
        the Ybus dimensions and non-singularity.

        Args:
            net: pandapower network (post-reconnection).
            result: ReconnectionResult to update with Ybus status.
        """
        # Need at least one bus and one line for Ybus
        if net.bus.empty or net.line.empty:
            msg = "Cannot validate Ybus: network has no buses or lines"
            result.warnings.append(msg)
            logger.warning(msg)
            return

        # Need at least one ext_grid for runpp
        if net.ext_grid.empty:
            msg = "Cannot validate Ybus: no ext_grid in network"
            result.warnings.append(msg)
            logger.warning(msg)
            return

        try:
            pp.runpp(net, numba=False)
        except Exception as exc:
            # Power flow may fail but _ppc may still be populated
            logger.debug(
                "pp.runpp() raised %s: %s (checking _ppc)",
                type(exc).__name__,
                exc,
            )

        # Extract Ybus
        if not hasattr(net, "_ppc") or net._ppc is None:
            msg = "Ybus validation failed: _ppc not populated after runpp"
            result.warnings.append(msg)
            logger.warning(msg)
            return

        internal = net._ppc.get("internal")
        if internal is None or "Ybus" not in internal:
            msg = "Ybus validation failed: Ybus not found in _ppc internal"
            result.warnings.append(msg)
            logger.warning(msg)
            return

        Ybus = internal["Ybus"]
        if not isinstance(Ybus, sparse.csc_matrix):
            Ybus = Ybus.tocsc()

        result.ybus_shape = Ybus.shape

        # Check non-singularity via diagonal dominance heuristic
        # A truly rigorous check would use sparse LU factorisation,
        # but checking that all diagonal entries are non-zero is a
        # fast practical proxy.
        diag = Ybus.diagonal()
        has_zero_diag = np.any(np.abs(diag) < 1e-12)
        result.ybus_nonsingular = not has_zero_diag

        if result.ybus_nonsingular:
            logger.info(
                "Ybus validation passed: shape=%s, all diagonal "
                "entries non-zero",
                result.ybus_shape,
            )
        else:
            n_zero = int(np.sum(np.abs(diag) < 1e-12))
            msg = (
                f"Ybus has {n_zero} near-zero diagonal entries — "
                f"matrix may be singular"
            )
            result.warnings.append(msg)
            logger.warning(msg)


# ------------------------------------------------------------------
# Module-level utilities
# ------------------------------------------------------------------


def _haversine_km(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
) -> float:
    """Compute Haversine distance between two (longitude, latitude) points.

    Args:
        point1: (longitude, latitude) in degrees.
        point2: (longitude, latitude) in degrees.

    Returns:
        Distance in kilometres.
    """
    lon1, lat1 = math.radians(point1[0]), math.radians(point1[1])
    lon2, lat2 = math.radians(point2[0]), math.radians(point2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    return _EARTH_RADIUS_KM * c
