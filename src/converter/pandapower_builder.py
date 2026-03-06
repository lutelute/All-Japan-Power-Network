"""Build pandapower network models from GridNetwork data.

Converts the pipeline's internal GridNetwork representation into a
pandapower network suitable for power flow analysis and MATPOWER export.

Handles Japan's dual-frequency system:
    - East Japan (Hokkaido, Tohoku, Tokyo): 50 Hz
    - West Japan (Chubu and westward): 60 Hz

For national (merged) models with ``frequency_hz=0``, the builder
defaults to 50 Hz but accepts a configurable override.

Optionally integrates with the reconstruction pipeline to handle
isolated network elements via simplification or reconnection before
producing the final pandapower network.

Usage::

    from src.converter.pandapower_builder import PandapowerBuilder

    builder = PandapowerBuilder()
    result = builder.build(grid_network)
    net = result.net
    # pp.runpp(net)  # Run power flow

    # With reconstruction:
    from src.reconstruction.config import ReconstructionConfig

    cfg = ReconstructionConfig(mode="simplify", seed=42)
    result = builder.build(grid_network, reconstruction_config=cfg)
    net = result.net  # Reconstructed network
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandapower as pp

from src.converter.line_parameters import get_line_parameters_safe
from src.model.grid_network import GridNetwork
from src.model.substation import BusType
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# Default frequency for national (mixed-frequency) networks
_DEFAULT_NATIONAL_FREQUENCY_HZ = 50

# East Japan regions operating at 50 Hz
_EAST_50HZ_REGIONS = frozenset({"hokkaido", "tohoku", "tokyo"})

# West Japan regions operating at 60 Hz
_WEST_60HZ_REGIONS = frozenset({
    "chubu", "hokuriku", "kansai", "chugoku", "shikoku", "kyushu", "okinawa",
})

# Region-to-frequency lookup for per-substation frequency resolution
_REGION_FREQUENCY_HZ: Dict[str, int] = {
    **{r: 50 for r in _EAST_50HZ_REGIONS},
    **{r: 60 for r in _WEST_60HZ_REGIONS},
}


@dataclass
class BuildResult:
    """Result of building a pandapower network from GridNetwork.

    Attributes:
        net: The constructed pandapower network.
        bus_map: Mapping from substation ID to pandapower bus index.
        line_map: Mapping from transmission line ID to pandapower line index.
        buses_created: Number of buses created.
        lines_created: Number of lines created.
        generators_created: Number of generators created.
        ext_grids_created: Number of external grid connections created.
        warnings: List of warning messages encountered during building.
        reconstruction_result: Optional result from the reconstruction
            pipeline.  Only set when ``build()`` is called with a
            ``reconstruction_config``.  The type is
            :class:`~src.reconstruction.pipeline.PipelineResult` but
            declared as ``Any`` to avoid circular imports.
    """

    net: Any  # pandapowerNet (avoid type annotation for portability)
    bus_map: Dict[str, int] = field(default_factory=dict)
    line_map: Dict[str, int] = field(default_factory=dict)
    buses_created: int = 0
    lines_created: int = 0
    generators_created: int = 0
    ext_grids_created: int = 0
    warnings: List[str] = field(default_factory=list)
    reconstruction_result: Optional[Any] = None

    @property
    def summary(self) -> Dict[str, object]:
        """Return a summary dict for logging."""
        result: Dict[str, object] = {
            "buses": self.buses_created,
            "lines": self.lines_created,
            "generators": self.generators_created,
            "ext_grids": self.ext_grids_created,
            "warnings": len(self.warnings),
        }
        if self.reconstruction_result is not None:
            result["reconstruction"] = self.reconstruction_result.summary
        return result


class PandapowerBuilder:
    """Builds pandapower network models from GridNetwork instances.

    Creates buses (from substations), lines (via ``create_line_from_parameters``
    with Japanese electrical parameters), generators (with ``vm_pu`` voltage
    setpoints), and external grid connections (slack bus) required for power
    flow convergence.

    When a :class:`~src.reconstruction.config.ReconstructionConfig` is
    supplied to :meth:`build`, the builder will additionally run the
    :class:`~src.reconstruction.pipeline.ReconstructionPipeline` on the
    constructed network, handling isolated elements via simplification
    or reconnection before returning the final result.

    Args:
        default_national_f_hz: Frequency to use for national (mixed) models
            where ``GridNetwork.frequency_hz == 0``. Defaults to 50 Hz.
    """

    def __init__(self, default_national_f_hz: int = _DEFAULT_NATIONAL_FREQUENCY_HZ) -> None:
        if default_national_f_hz not in (50, 60):
            raise ValueError(
                f"default_national_f_hz must be 50 or 60, got {default_national_f_hz}"
            )
        self._default_national_f_hz = default_national_f_hz

    def build(
        self,
        network: GridNetwork,
        reconstruction_config: Optional[Any] = None,
    ) -> BuildResult:
        """Build a pandapower network from a GridNetwork.

        Processes substations → buses, transmission lines → lines,
        generators → gen elements, and designates a slack bus with
        ``create_ext_grid()``.

        When *reconstruction_config* is provided, the constructed
        network is passed through the
        :class:`~src.reconstruction.pipeline.ReconstructionPipeline`
        to handle isolated elements (simplification or reconnection)
        and synthesise missing load/generation data.  The pipeline
        result is stored in
        :attr:`BuildResult.reconstruction_result`.

        Args:
            network: The GridNetwork instance to convert.
            reconstruction_config: Optional reconstruction configuration
                (:class:`~src.reconstruction.config.ReconstructionConfig`).
                When ``None`` (default), no reconstruction is performed
                and the builder behaves identically to previous versions.

        Returns:
            BuildResult containing the pandapower network and metadata.

        Raises:
            ValueError: If the network has no substations.
        """
        if not network.substations:
            raise ValueError(
                f"Cannot build pandapower network for region "
                f"'{network.region}': no substations"
            )

        f_hz = self._resolve_frequency(network)
        net = pp.create_empty_network(
            name=f"japan_grid_{network.region}",
            f_hz=f_hz,
        )

        logger.info(
            "Building pandapower network for '%s' (f=%d Hz, "
            "%d substations, %d lines, %d generators)",
            network.region,
            f_hz,
            network.substation_count,
            network.line_count,
            network.generator_count,
        )

        result = BuildResult(net=net)

        # Step 1: Create buses from substations
        self._create_buses(net, network, result)

        # Step 2: Create lines from transmission lines
        self._create_lines(net, network, f_hz, result)

        # Step 3: Create generators
        self._create_generators(net, network, result)

        # Step 4: Create external grid (slack bus)
        self._create_ext_grid(net, network, result)

        # Step 5: Infer bus voltages from connected line voltages
        self._infer_bus_voltages(net, network, result)

        # Step 6 (optional): Run reconstruction pipeline
        if reconstruction_config is not None:
            self._run_reconstruction(net, network, reconstruction_config, result)

        logger.info(
            "Built pandapower network '%s': %s",
            net.name,
            result.summary,
        )

        return result

    # ------------------------------------------------------------------
    # Internal: bus creation
    # ------------------------------------------------------------------

    def _create_buses(
        self,
        net: Any,
        network: GridNetwork,
        result: BuildResult,
    ) -> None:
        """Create pandapower buses from substations.

        Each substation becomes a bus with its nominal voltage and
        geographic coordinates. The substation's region is stored in
        the bus table ``zone`` column for traceability.

        Args:
            net: The pandapower network.
            network: Source GridNetwork.
            result: BuildResult to update with bus_map and counts.
        """
        for sub in network.substations:
            bus_idx = pp.create_bus(
                net,
                vn_kv=sub.voltage_kv,
                name=sub.name,
                geodata=sub.geodata,
            )

            # Set zone after creation (not a create_bus parameter)
            net.bus.at[bus_idx, "zone"] = sub.region

            result.bus_map[sub.id] = bus_idx

        result.buses_created = len(result.bus_map)

        logger.debug(
            "Created %d buses for region '%s'",
            result.buses_created,
            network.region,
        )

    # ------------------------------------------------------------------
    # Internal: line creation
    # ------------------------------------------------------------------

    def _create_lines(
        self,
        net: Any,
        network: GridNetwork,
        f_hz: int,
        result: BuildResult,
    ) -> None:
        """Create pandapower lines from transmission lines.

        Uses ``create_line_from_parameters()`` with Japanese electrical
        parameters. If a transmission line lacks electrical parameters,
        they are looked up from the reference table via
        ``get_line_parameters_safe()``.

        Args:
            net: The pandapower network.
            network: Source GridNetwork.
            f_hz: Network frequency in Hz for parameter lookup.
            result: BuildResult to update with counts and warnings.
        """
        lines_created = 0
        lines_skipped = 0

        for line in network.lines:
            # Resolve bus indices
            from_bus = result.bus_map.get(line.from_substation_id)
            to_bus = result.bus_map.get(line.to_substation_id)

            if from_bus is None:
                msg = (
                    f"Line '{line.id}': from_substation "
                    f"'{line.from_substation_id}' not found in bus_map"
                )
                result.warnings.append(msg)
                logger.warning(msg)
                lines_skipped += 1
                continue

            if to_bus is None:
                msg = (
                    f"Line '{line.id}': to_substation "
                    f"'{line.to_substation_id}' not found in bus_map"
                )
                result.warnings.append(msg)
                logger.warning(msg)
                lines_skipped += 1
                continue

            # Skip zero-length lines (degenerate)
            if line.length_km <= 0:
                msg = (
                    f"Line '{line.id}': skipping zero-length line "
                    f"({line.length_km} km)"
                )
                result.warnings.append(msg)
                logger.warning(msg)
                lines_skipped += 1
                continue

            # Resolve electrical parameters
            r_ohm, x_ohm, c_nf, max_i = self._resolve_line_params(
                line, f_hz, result,
            )

            # Ensure minimum reactance to avoid numerical issues
            if x_ohm <= 0:
                x_ohm = 0.001
                msg = (
                    f"Line '{line.id}': zero reactance replaced "
                    f"with minimum 0.001 Ohm/km"
                )
                result.warnings.append(msg)
                logger.warning(msg)

            line_idx = pp.create_line_from_parameters(
                net,
                from_bus=from_bus,
                to_bus=to_bus,
                length_km=line.length_km,
                r_ohm_per_km=r_ohm,
                x_ohm_per_km=x_ohm,
                c_nf_per_km=c_nf,
                max_i_ka=max_i,
                name=line.name,
            )

            result.line_map[line.id] = line_idx
            lines_created += 1

        result.lines_created = lines_created

        if lines_skipped > 0:
            logger.warning(
                "Skipped %d lines (missing endpoints or zero length)",
                lines_skipped,
            )

        logger.debug(
            "Created %d lines for region '%s'",
            lines_created,
            network.region,
        )

    def _resolve_line_params(
        self,
        line: Any,
        f_hz: int,
        result: BuildResult,
    ) -> tuple:
        """Resolve electrical parameters for a transmission line.

        Uses the line's own parameters if available, otherwise falls
        back to the reference table via ``get_line_parameters_safe()``.

        Args:
            line: TransmissionLine instance.
            f_hz: Network frequency for parameter lookup.
            result: BuildResult for accumulating warnings.

        Returns:
            Tuple of (r_ohm_per_km, x_ohm_per_km, c_nf_per_km, max_i_ka).
        """
        if line.has_electrical_parameters:
            return (
                line.r_ohm_per_km,
                line.x_ohm_per_km,
                line.c_nf_per_km,
                line.max_i_ka,
            )

        # Fall back to reference table
        ref_params = get_line_parameters_safe(line.voltage_kv, f_hz)

        if ref_params is not None:
            logger.debug(
                "Line '%s': using reference parameters for %.0f kV @ %d Hz",
                line.id,
                line.voltage_kv,
                f_hz,
            )
            return (
                ref_params["r_ohm_per_km"],
                ref_params["x_ohm_per_km"],
                ref_params["c_nf_per_km"],
                ref_params["max_i_ka"],
            )

        # Last resort: generic defaults
        msg = (
            f"Line '{line.id}': no parameters for {line.voltage_kv} kV; "
            f"using generic defaults"
        )
        result.warnings.append(msg)
        logger.warning(msg)

        return (0.05, 0.4, 10.0, 1.0)

    # ------------------------------------------------------------------
    # Internal: generator creation
    # ------------------------------------------------------------------

    def _create_generators(
        self,
        net: Any,
        network: GridNetwork,
        result: BuildResult,
    ) -> None:
        """Create pandapower generators from Generator instances.

        Each connected generator becomes a ``gen`` element with its
        capacity and voltage setpoint (``vm_pu``). Unconnected generators
        (no ``connected_bus_id``) are skipped with a warning.

        Args:
            net: The pandapower network.
            network: Source GridNetwork.
            result: BuildResult to update with counts and warnings.
        """
        gens_created = 0
        gens_skipped = 0

        for gen in network.generators:
            if not gen.connected_bus_id:
                msg = (
                    f"Generator '{gen.id}' ({gen.name}): "
                    f"not connected to any bus, skipping"
                )
                result.warnings.append(msg)
                logger.warning(msg)
                gens_skipped += 1
                continue

            bus_idx = result.bus_map.get(gen.connected_bus_id)
            if bus_idx is None:
                msg = (
                    f"Generator '{gen.id}': connected_bus "
                    f"'{gen.connected_bus_id}' not found in bus_map"
                )
                result.warnings.append(msg)
                logger.warning(msg)
                gens_skipped += 1
                continue

            pp.create_gen(
                net,
                bus=bus_idx,
                p_mw=gen.capacity_mw,
                vm_pu=gen.vm_pu,
                name=gen.name,
                min_p_mw=gen.p_min_mw,
                max_p_mw=gen.capacity_mw,
            )

            gens_created += 1

        result.generators_created = gens_created

        if gens_skipped > 0:
            logger.warning(
                "Skipped %d generators (unconnected or missing bus)",
                gens_skipped,
            )

        logger.debug(
            "Created %d generators for region '%s'",
            gens_created,
            network.region,
        )

    # ------------------------------------------------------------------
    # Internal: external grid (slack bus)
    # ------------------------------------------------------------------

    def _create_ext_grid(
        self,
        net: Any,
        network: GridNetwork,
        result: BuildResult,
    ) -> None:
        """Create external grid connection(s) on the slack bus.

        Every pandapower network requires at least one ``ext_grid``
        element (slack/reference bus) for power flow convergence.
        The slack bus is selected by:
            1. The substation explicitly marked as ``BusType.SLACK``.
            2. The bus connected to the highest-capacity generator.
            3. The first bus in the network (fallback).

        Args:
            net: The pandapower network.
            network: Source GridNetwork.
            result: BuildResult to update with counts.
        """
        slack_bus_idx = self._determine_slack_bus(network, result)

        pp.create_ext_grid(
            net,
            bus=slack_bus_idx,
            vm_pu=1.0,
            name="slack_grid",
        )

        result.ext_grids_created = 1

        # Identify the substation name for logging
        slack_name = net.bus.at[slack_bus_idx, "name"]
        logger.info(
            "Created ext_grid (slack bus) at bus %d ('%s') for region '%s'",
            slack_bus_idx,
            slack_name,
            network.region,
        )

    def _determine_slack_bus(
        self,
        network: GridNetwork,
        result: BuildResult,
    ) -> int:
        """Determine the pandapower bus index for the slack bus.

        Selection priority:
            1. Substation explicitly marked as ``BusType.SLACK``.
            2. Bus connected to the highest-capacity generator.
            3. First bus in the network.

        Args:
            network: Source GridNetwork.
            result: BuildResult containing bus_map.

        Returns:
            pandapower bus index for the slack bus.
        """
        # Priority 1: Explicitly marked slack substations
        for sub in network.substations:
            if sub.bus_type == BusType.SLACK.value:
                bus_idx = result.bus_map.get(sub.id)
                if bus_idx is not None:
                    logger.debug(
                        "Slack bus selected: explicit SLACK substation '%s'",
                        sub.name,
                    )
                    return bus_idx

        # Priority 2: Bus with highest-capacity generator
        if network.generators:
            connected_gens = [
                gen for gen in network.generators
                if gen.connected_bus_id and gen.connected_bus_id in result.bus_map
            ]
            if connected_gens:
                best_gen = max(connected_gens, key=lambda g: g.capacity_mw)
                bus_idx = result.bus_map[best_gen.connected_bus_id]
                logger.debug(
                    "Slack bus selected: highest-capacity generator '%s' "
                    "(%.1f MW) at bus %d",
                    best_gen.name,
                    best_gen.capacity_mw,
                    bus_idx,
                )
                return bus_idx

        # Priority 3: First bus
        first_sub_id = network.substations[0].id
        bus_idx = result.bus_map[first_sub_id]
        msg = (
            f"No SLACK substation or generators found; "
            f"using first bus '{network.substations[0].name}' as slack"
        )
        result.warnings.append(msg)
        logger.warning(msg)

        return bus_idx

    # ------------------------------------------------------------------
    # Internal: bus voltage inference
    # ------------------------------------------------------------------

    def _infer_bus_voltages(
        self,
        net: Any,
        network: GridNetwork,
        result: BuildResult,
    ) -> None:
        """Infer bus voltages from connected transmission line voltages.

        Many KML substations lack voltage information, resulting in
        ``vn_kv=0``.  This causes division by zero in pandapower's
        per-unit calculation.  For each zero-voltage bus, we set its
        voltage to the highest voltage of any connected line.  Remaining
        zero-voltage buses get the median voltage of the network.

        Args:
            net: The pandapower network.
            network: Source GridNetwork (for line voltage data).
            result: BuildResult (for bus_map).
        """
        zero_mask = net.bus["vn_kv"] == 0
        n_zero = int(zero_mask.sum())
        if n_zero == 0:
            return

        # Build bus_index → max line voltage mapping
        bus_voltages: Dict[int, float] = {}
        for line in network.lines:
            if line.voltage_kv <= 0:
                continue
            for sub_id in (line.from_substation_id, line.to_substation_id):
                bus_idx = result.bus_map.get(sub_id)
                if bus_idx is not None:
                    if bus_idx not in bus_voltages or line.voltage_kv > bus_voltages[bus_idx]:
                        bus_voltages[bus_idx] = line.voltage_kv

        fixed = 0
        for bus_idx, voltage in bus_voltages.items():
            if bus_idx in net.bus.index and net.bus.at[bus_idx, "vn_kv"] == 0:
                net.bus.at[bus_idx, "vn_kv"] = voltage
                fixed += 1

        # Fallback: remaining zero-voltage buses get median
        still_zero = net.bus["vn_kv"] == 0
        if still_zero.any():
            nonzero = net.bus.loc[~still_zero, "vn_kv"]
            if len(nonzero) > 0:
                median_kv = float(nonzero.median())
                net.bus.loc[still_zero, "vn_kv"] = median_kv
                fixed += int(still_zero.sum())

        logger.info(
            "Voltage inference: fixed %d/%d zero-voltage buses",
            fixed,
            n_zero,
        )

    # ------------------------------------------------------------------
    # Internal: reconstruction pipeline integration
    # ------------------------------------------------------------------

    def _run_reconstruction(
        self,
        net: Any,
        network: GridNetwork,
        reconstruction_config: Any,
        result: BuildResult,
    ) -> None:
        """Run the reconstruction pipeline on the built network.

        Lazily imports the reconstruction pipeline to avoid circular
        dependencies and keep reconstruction an optional feature.

        The pipeline modifies the network in place (``copy_network=False``
        since the builder already owns the network) and the result is
        stored in ``result.reconstruction_result``.

        Args:
            net: The pandapower network built in previous steps.
            network: Source GridNetwork (used for region identifier).
            reconstruction_config: A
                :class:`~src.reconstruction.config.ReconstructionConfig`
                instance controlling the reconstruction mode, seed,
                and thresholds.
            result: BuildResult to update with reconstruction results
                and warnings.
        """
        # Lazy import to avoid circular dependencies and keep
        # reconstruction optional for callers that don't need it.
        from src.reconstruction.pipeline import ReconstructionPipeline

        logger.info(
            "Running reconstruction pipeline (mode=%s, seed=%d) "
            "on network '%s'",
            reconstruction_config.mode,
            reconstruction_config.seed,
            net.name,
        )

        pipeline = ReconstructionPipeline(
            config=reconstruction_config,
            copy_network=False,  # Builder owns the network; modify in place
        )

        pipeline_result = pipeline.run(net, region=network.region)

        result.reconstruction_result = pipeline_result

        # Update the net reference in case reconstruction replaced it
        # (e.g. the pipeline deep-copied internally despite our setting)
        result.net = pipeline_result.net

        # Propagate reconstruction warnings to the build result
        result.warnings.extend(pipeline_result.warnings)

        logger.info(
            "Reconstruction pipeline complete for '%s': %s",
            net.name,
            pipeline_result.summary,
        )

    # ------------------------------------------------------------------
    # Frequency resolution
    # ------------------------------------------------------------------

    def _resolve_frequency(self, network: GridNetwork) -> int:
        """Resolve the network frequency for pandapower.

        Regional networks use their defined frequency. National
        (merged) networks with ``frequency_hz=0`` use the configured
        default (typically 50 Hz).

        Args:
            network: Source GridNetwork.

        Returns:
            Frequency in Hz (50 or 60).
        """
        if network.frequency_hz in (50, 60):
            return network.frequency_hz

        # National / mixed model
        logger.info(
            "Network '%s' has frequency_hz=%d (mixed/national); "
            "using default %d Hz",
            network.region,
            network.frequency_hz,
            self._default_national_f_hz,
        )
        return self._default_national_f_hz

    # ------------------------------------------------------------------
    # Convenience: frequency lookup for individual elements
    # ------------------------------------------------------------------

    @staticmethod
    def get_region_frequency(region: str) -> int:
        """Look up the system frequency for a given region.

        Args:
            region: Region identifier (e.g., 'hokkaido', 'chubu').

        Returns:
            Frequency in Hz (50 or 60).

        Raises:
            ValueError: If the region is not recognized.
        """
        f_hz = _REGION_FREQUENCY_HZ.get(region)
        if f_hz is None:
            raise ValueError(
                f"Unknown region '{region}'. Expected one of: "
                f"{sorted(_REGION_FREQUENCY_HZ.keys())}"
            )
        return f_hz
