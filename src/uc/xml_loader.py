"""Load generator data from standardized XML into UC-ready format.

Parses generator elements from the Japan Grid standardized XML
(``schemas/power_grid.xsd``), extracts UC-specific attributes, and
applies configurable defaults for any missing optional fields. Produces
a list of :class:`~src.model.generator.Generator` objects ready for
unit commitment solving.

Supports three levels of input data:
  - **Complete data**: All UC attributes present in XML.
  - **Partial data**: Some UC fields present, remainder filled from defaults.
  - **Basic data**: No UC fields at all; all UC attributes from defaults.

Usage::

    from src.uc.xml_loader import UCXMLLoader

    loader = UCXMLLoader()
    generators = loader.load_generators_from_xml(
        "output/xml/japan_grid_all.xml",
        "config/uc_config.yaml",
    )
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml
from lxml import etree

from src.model.generator import Generator
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# XML namespace for the Japan Grid schema
NAMESPACE = "urn:japan-grid:v1"

# Base datetime used by xml_exporter for maintenance window serialization
_MAINTENANCE_BASE_DT = datetime(2000, 1, 1)

# Default path to the UC configuration file (relative to project root)
_DEFAULT_CONFIG_PATH = os.path.join("config", "uc_config.yaml")


def _ns(tag: str) -> str:
    """Return a namespace-qualified tag name.

    Args:
        tag: Local tag name (e.g., ``"Generator"``).

    Returns:
        Fully qualified tag string (e.g., ``"{urn:japan-grid:v1}Generator"``).
    """
    return f"{{{NAMESPACE}}}{tag}"


def _get_attr_float(
    elem: etree._Element, attr: str, default: Optional[float] = None
) -> Optional[float]:
    """Extract a float attribute from an XML element.

    Args:
        elem: lxml Element to read from.
        attr: Attribute name.
        default: Value to return if the attribute is absent.

    Returns:
        Parsed float value, or *default* if the attribute is missing.
    """
    val = elem.get(attr)
    if val is None:
        return default
    return float(val)


def _get_attr_int(
    elem: etree._Element, attr: str, default: Optional[int] = None
) -> Optional[int]:
    """Extract an integer attribute from an XML element.

    Args:
        elem: lxml Element to read from.
        attr: Attribute name.
        default: Value to return if the attribute is absent.

    Returns:
        Parsed integer value, or *default* if the attribute is missing.
    """
    val = elem.get(attr)
    if val is None:
        return default
    return int(float(val))


def _get_attr_str(
    elem: etree._Element, attr: str, default: str = ""
) -> str:
    """Extract a string attribute from an XML element.

    Args:
        elem: lxml Element to read from.
        attr: Attribute name.
        default: Value to return if the attribute is absent.

    Returns:
        Attribute value as string, or *default* if missing.
    """
    val = elem.get(attr)
    if val is None:
        return default
    return val


class UCXMLLoader:
    """Load generators from standardized XML with UC attribute defaults.

    Parses generator elements from XML files conforming to the
    ``power_grid.xsd`` schema, extracts UC-specific attributes, and
    fills missing optional fields from the ``defaults`` section of
    ``uc_config.yaml``.

    Attributes:
        defaults: Dictionary of default UC attribute values loaded
            from the configuration file.
    """

    def __init__(self) -> None:
        """Initialize the UCXMLLoader."""
        self.defaults: Dict[str, Any] = {}
        logger.info("UCXMLLoader initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_generators_from_xml(
        self,
        xml_path: str,
        config_path: Optional[str] = None,
    ) -> List[Generator]:
        """Parse generator elements from XML and return UC-ready Generators.

        Reads all ``<Generator>`` elements across all ``<Region>`` elements
        in the XML document. For each generator, extracts required attributes
        (id, name, capacity_mw, fuel_type) and optional UC attributes.
        Missing UC attributes are filled from the defaults section of the
        UC configuration file.

        Args:
            xml_path: Path to the standardized XML file.
            config_path: Path to the UC configuration YAML file. Defaults
                to ``config/uc_config.yaml``.

        Returns:
            List of :class:`Generator` objects with all UC fields populated.

        Raises:
            FileNotFoundError: If *xml_path* or *config_path* does not exist.
            etree.XMLSyntaxError: If the XML file is malformed.
        """
        if config_path is None:
            config_path = _DEFAULT_CONFIG_PATH

        # Load configuration defaults
        self._load_config(config_path)

        # Parse XML
        logger.info("Loading generators from XML: %s", xml_path)
        tree = etree.parse(xml_path)
        root = tree.getroot()

        generators: List[Generator] = []

        # Find all Generator elements across all Regions
        for region_elem in root.iter(_ns("Region")):
            region_id = region_elem.get("id", "")
            generators_elem = region_elem.find(_ns("Generators"))
            if generators_elem is None:
                continue

            for gen_elem in generators_elem.findall(_ns("Generator")):
                gen = self._parse_generator(gen_elem, region_id)
                generators.append(gen)

        logger.info(
            "Loaded %d generators from XML (config defaults from %s)",
            len(generators),
            config_path,
        )
        return generators

    # ------------------------------------------------------------------
    # Private: Configuration
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> None:
        """Load UC defaults from the configuration YAML file.

        Args:
            config_path: Path to the UC configuration YAML file.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        logger.debug("Loading UC config defaults from: %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.defaults = cfg.get("defaults", {})
        logger.debug(
            "UC defaults loaded: %s",
            list(self.defaults.keys()),
        )

    # ------------------------------------------------------------------
    # Private: Generator Parsing
    # ------------------------------------------------------------------

    def _parse_generator(
        self,
        elem: etree._Element,
        region_id: str,
    ) -> Generator:
        """Parse a single Generator XML element into a Generator object.

        Extracts all required and optional attributes from the XML element.
        For missing UC-specific attributes, applies defaults from the
        loaded configuration.

        Args:
            elem: lxml Element representing a ``<Generator>``.
            region_id: Region identifier from the parent ``<Region>`` element.

        Returns:
            Fully populated :class:`Generator` instance.
        """
        # --- Required attributes ---
        gen_id = elem.get("id", "")
        name = elem.get("name", "")
        capacity_mw = float(elem.get("capacity_mw", "0"))
        fuel_type = elem.get("fuel_type", "unknown")
        status = _get_attr_str(elem, "status", "active")
        connected_bus_id = _get_attr_str(elem, "connected_substation", "")

        # --- Location (optional child element) ---
        latitude = 0.0
        longitude = 0.0
        location_elem = elem.find(_ns("Location"))
        if location_elem is not None:
            latitude = _get_attr_float(location_elem, "latitude", 0.0)
            longitude = _get_attr_float(location_elem, "longitude", 0.0)

        # --- UC attributes (with config defaults fallback) ---
        startup_cost = self._get_uc_float(elem, "startup_cost", fuel_type)
        shutdown_cost = self._get_uc_float(elem, "shutdown_cost", fuel_type)
        min_up_time_h = self._get_uc_int(elem, "min_up_time_h", fuel_type)
        min_down_time_h = self._get_uc_int(elem, "min_down_time_h", fuel_type)
        fuel_cost_per_mwh = self._get_fuel_cost(elem, fuel_type)
        labor_cost_per_h = self._get_uc_float(elem, "labor_cost_per_h", fuel_type)
        no_load_cost = self._get_uc_float(elem, "no_load_cost", fuel_type)

        # --- RampRates (optional child element) ---
        ramp_up_mw_per_h = self.defaults.get("ramp_up_mw_per_h")
        ramp_down_mw_per_h = self.defaults.get("ramp_down_mw_per_h")
        ramp_elem = elem.find(_ns("RampRates"))
        if ramp_elem is not None:
            ramp_up_val = _get_attr_float(ramp_elem, "ramp_up_mw_per_h")
            if ramp_up_val is not None:
                ramp_up_mw_per_h = ramp_up_val
            ramp_down_val = _get_attr_float(ramp_elem, "ramp_down_mw_per_h")
            if ramp_down_val is not None:
                ramp_down_mw_per_h = ramp_down_val

        # --- MaintenancePlan (optional child element) ---
        maintenance_windows = self._parse_maintenance_plan(elem)

        # --- RebuildPlan (optional child element) ---
        rebuild_planned_date = self._parse_rebuild_plan(elem)

        # --- DisasterRisk (optional child element) ---
        disaster_risk_score = self._parse_disaster_risk(elem)

        gen = Generator(
            id=gen_id,
            name=name,
            capacity_mw=capacity_mw,
            fuel_type=fuel_type,
            connected_bus_id=connected_bus_id,
            region=region_id,
            latitude=latitude,
            longitude=longitude,
            status=status,
            startup_cost=startup_cost,
            shutdown_cost=shutdown_cost,
            min_up_time_h=min_up_time_h,
            min_down_time_h=min_down_time_h,
            ramp_up_mw_per_h=ramp_up_mw_per_h,
            ramp_down_mw_per_h=ramp_down_mw_per_h,
            fuel_cost_per_mwh=fuel_cost_per_mwh,
            labor_cost_per_h=labor_cost_per_h,
            no_load_cost=no_load_cost,
            maintenance_windows=maintenance_windows,
            rebuild_planned_date=rebuild_planned_date,
            disaster_risk_score=disaster_risk_score,
        )

        logger.debug(
            "Parsed generator '%s' (%s, %.1f MW, fuel=%s)",
            gen_id,
            name,
            capacity_mw,
            fuel_type,
        )
        return gen

    # ------------------------------------------------------------------
    # Private: UC Attribute Extraction with Defaults
    # ------------------------------------------------------------------

    def _get_uc_float(
        self,
        elem: etree._Element,
        attr: str,
        fuel_type: str,
    ) -> float:
        """Extract a UC float attribute from XML, falling back to config default.

        Args:
            elem: Generator XML element.
            attr: Attribute name (e.g., ``"startup_cost"``).
            fuel_type: Fuel type string for fuel-type-specific defaults.

        Returns:
            Float value from XML attribute, or from config defaults.
        """
        val = elem.get(attr)
        if val is not None:
            return float(val)
        # Fall back to config default
        default_val = self.defaults.get(attr, 0.0)
        return float(default_val) if default_val is not None else 0.0

    def _get_uc_int(
        self,
        elem: etree._Element,
        attr: str,
        fuel_type: str,
    ) -> int:
        """Extract a UC integer attribute from XML, falling back to config default.

        Args:
            elem: Generator XML element.
            attr: Attribute name (e.g., ``"min_up_time_h"``).
            fuel_type: Fuel type string for fuel-type-specific defaults.

        Returns:
            Integer value from XML attribute, or from config defaults.
        """
        val = elem.get(attr)
        if val is not None:
            return int(float(val))
        # Fall back to config default
        default_val = self.defaults.get(attr, 1)
        return int(default_val) if default_val is not None else 1

    def _get_fuel_cost(
        self,
        elem: etree._Element,
        fuel_type: str,
    ) -> float:
        """Extract fuel cost per MWh, with fuel-type-specific default.

        Checks the XML ``fuel_cost_per_mwh`` attribute first. If absent,
        looks up the fuel-type-specific default from the config's
        ``fuel_cost_per_mwh`` mapping. Falls back to 0.0 if neither is
        available.

        Args:
            elem: Generator XML element.
            fuel_type: Fuel type string for lookup in defaults map.

        Returns:
            Fuel cost per MWh as a float.
        """
        val = elem.get("fuel_cost_per_mwh")
        if val is not None:
            return float(val)

        # Look up fuel-type-specific default
        fuel_cost_map = self.defaults.get("fuel_cost_per_mwh", {})
        if isinstance(fuel_cost_map, dict):
            fuel_cost = fuel_cost_map.get(fuel_type)
            if fuel_cost is not None:
                return float(fuel_cost)

        # Fallback for unknown fuel types
        if isinstance(fuel_cost_map, dict):
            unknown_cost = fuel_cost_map.get("unknown")
            if unknown_cost is not None:
                logger.warning(
                    "No fuel_cost_per_mwh default for fuel_type '%s', "
                    "using 'unknown' default",
                    fuel_type,
                )
                return float(unknown_cost)

        return 0.0

    # ------------------------------------------------------------------
    # Private: Child Element Parsers
    # ------------------------------------------------------------------

    def _parse_maintenance_plan(
        self,
        gen_elem: etree._Element,
    ) -> List[Tuple[int, int]]:
        """Parse MaintenancePlan child element into (start_h, end_h) tuples.

        The XML stores maintenance windows as ISO 8601 datetime strings
        relative to a base datetime of 2000-01-01T00:00:00. This method
        converts them back to integer hour offsets.

        Args:
            gen_elem: Generator XML element.

        Returns:
            List of (start_hour, end_hour) tuples. Empty list if no
            MaintenancePlan is present.
        """
        mp_elem = gen_elem.find(_ns("MaintenancePlan"))
        if mp_elem is None:
            return []

        windows: List[Tuple[int, int]] = []
        for window_elem in mp_elem.findall(_ns("Window")):
            start_str = window_elem.get("start")
            end_str = window_elem.get("end")
            if start_str is None or end_str is None:
                logger.warning(
                    "Skipping maintenance window with missing start/end"
                )
                continue

            try:
                start_dt = datetime.fromisoformat(start_str)
                end_dt = datetime.fromisoformat(end_str)
                start_h = int(
                    (start_dt - _MAINTENANCE_BASE_DT).total_seconds() / 3600
                )
                end_h = int(
                    (end_dt - _MAINTENANCE_BASE_DT).total_seconds() / 3600
                )
                if start_h < end_h:
                    windows.append((start_h, end_h))
                else:
                    logger.warning(
                        "Skipping invalid maintenance window: "
                        "start_h=%d >= end_h=%d",
                        start_h,
                        end_h,
                    )
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Skipping maintenance window with invalid datetime: %s",
                    exc,
                )

        return windows

    def _parse_rebuild_plan(
        self,
        gen_elem: etree._Element,
    ) -> Optional[str]:
        """Parse RebuildPlan child element for the planned date.

        Args:
            gen_elem: Generator XML element.

        Returns:
            Planned date as ISO 8601 string, or ``None`` if absent.
        """
        rp_elem = gen_elem.find(_ns("RebuildPlan"))
        if rp_elem is None:
            return None

        planned_date = rp_elem.get("planned_date")
        return planned_date if planned_date else None

    def _parse_disaster_risk(
        self,
        gen_elem: etree._Element,
    ) -> float:
        """Parse DisasterRisk child element for the risk score.

        Falls back to the config default if the element is absent.

        Args:
            gen_elem: Generator XML element.

        Returns:
            Disaster risk score as a float.
        """
        dr_elem = gen_elem.find(_ns("DisasterRisk"))
        if dr_elem is not None:
            risk_score = _get_attr_float(dr_elem, "risk_score")
            if risk_score is not None:
                return risk_score

        # Fall back to config default
        default_val = self.defaults.get("disaster_risk_score", 0.0)
        return float(default_val) if default_val is not None else 0.0
