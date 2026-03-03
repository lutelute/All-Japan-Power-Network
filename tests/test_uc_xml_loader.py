"""Unit tests for the UCXMLLoader module.

Tests XML loading of generator data with UC-specific attributes, including:
- Loading generators from well-formed XML with all UC attributes
- Loading from XML with only basic attributes (UC fields get defaults)
- Loading from XML with partial UC attributes
- Error handling for malformed XML and missing files
- Config defaults application (fuel-type-specific and generic)
"""

import textwrap
from pathlib import Path
from typing import Dict, List

import pytest
import yaml
from lxml import etree

from src.model.generator import Generator
from src.uc.xml_loader import (
    NAMESPACE,
    UCXMLLoader,
    _get_attr_float,
    _get_attr_int,
    _get_attr_str,
    _ns,
)


# ======================================================================
# XML fixture builders
# ======================================================================

_MINIMAL_CONFIG = {
    "defaults": {
        "startup_cost": 5000,
        "shutdown_cost": 2000,
        "min_up_time_h": 4,
        "min_down_time_h": 4,
        "ramp_up_mw_per_h": None,
        "ramp_down_mw_per_h": None,
        "fuel_cost_per_mwh": {
            "coal": 4500,
            "lng": 7000,
            "oil": 9000,
            "nuclear": 1500,
            "hydro": 0,
            "pumped_hydro": 2000,
            "geothermal": 0,
            "wind": 0,
            "solar": 0,
            "biomass": 3000,
            "mixed": 5000,
            "unknown": 5000,
        },
        "labor_cost_per_h": 1000,
        "no_load_cost": 500,
        "disaster_risk_score": 0.0,
    }
}


def _build_xml(generators_xml: str, region_id: str = "shikoku") -> str:
    """Build a complete power grid XML document with given generator fragments.

    Args:
        generators_xml: XML fragment for Generator elements.
        region_id: Region id for the wrapping Region element.

    Returns:
        Complete XML string conforming to the power_grid.xsd schema structure.
    """
    # Build template separately to avoid textwrap.dedent interaction with
    # interpolated content (which can leave whitespace before <?xml>).
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PowerGrid xmlns="urn:japan-grid:v1" version="1.0" created="2026-01-01">\n'
        "  <Metadata>\n"
        "    <Name>Test Grid</Name>\n"
        "    <Description>Test</Description>\n"
        "    <Source>Test</Source>\n"
        "    <Attribution>Test</Attribution>\n"
        "    <RegionCount>1</RegionCount>\n"
        "  </Metadata>\n"
        "  <Regions>\n"
        f'    <Region id="{region_id}" name="Test Region" frequency_hz="60">\n'
        "      <Substations/>\n"
        "      <TransmissionLines/>\n"
        "      <Generators>\n"
        f"        {generators_xml}\n"
        "      </Generators>\n"
        "    </Region>\n"
        "  </Regions>\n"
        "</PowerGrid>\n"
    )


def _write_xml(tmp_path: Path, xml_content: str, filename: str = "test.xml") -> str:
    """Write XML content to a temporary file and return the path string.

    Args:
        tmp_path: pytest tmp_path fixture.
        xml_content: XML string to write.
        filename: Output filename.

    Returns:
        Absolute path to the written file.
    """
    xml_file = tmp_path / filename
    xml_file.write_text(xml_content, encoding="utf-8")
    return str(xml_file)


def _write_config(tmp_path: Path, config: Dict = None) -> str:
    """Write a UC config YAML to a temporary file and return the path string.

    Args:
        tmp_path: pytest tmp_path fixture.
        config: Configuration dictionary. Defaults to _MINIMAL_CONFIG.

    Returns:
        Absolute path to the written config file.
    """
    if config is None:
        config = _MINIMAL_CONFIG
    config_file = tmp_path / "uc_config.yaml"
    config_file.write_text(yaml.dump(config), encoding="utf-8")
    return str(config_file)


# ======================================================================
# Helper function tests
# ======================================================================


class TestNsHelper:
    """Tests for the _ns namespace helper function."""

    def test_ns_produces_qualified_tag(self) -> None:
        """_ns wraps a local tag name with the namespace."""
        result = _ns("Generator")
        assert result == f"{{{NAMESPACE}}}Generator"

    def test_ns_empty_tag(self) -> None:
        """_ns handles an empty tag name."""
        result = _ns("")
        assert result == f"{{{NAMESPACE}}}"


class TestGetAttrFloat:
    """Tests for the _get_attr_float helper function."""

    def test_present_value(self) -> None:
        """Returns parsed float when attribute is present."""
        elem = etree.Element("test", capacity="100.5")
        assert _get_attr_float(elem, "capacity") == 100.5

    def test_missing_returns_default(self) -> None:
        """Returns default when attribute is absent."""
        elem = etree.Element("test")
        assert _get_attr_float(elem, "capacity", 42.0) == 42.0

    def test_missing_returns_none_default(self) -> None:
        """Returns None when attribute is absent and default is None."""
        elem = etree.Element("test")
        assert _get_attr_float(elem, "capacity") is None

    def test_negative_value(self) -> None:
        """Correctly parses negative float values."""
        elem = etree.Element("test", cost="-10.5")
        assert _get_attr_float(elem, "cost") == -10.5


class TestGetAttrInt:
    """Tests for the _get_attr_int helper function."""

    def test_present_integer(self) -> None:
        """Returns parsed int when attribute is an integer string."""
        elem = etree.Element("test", hours="4")
        assert _get_attr_int(elem, "hours") == 4

    def test_present_float_string(self) -> None:
        """Returns truncated int when attribute is a float string."""
        elem = etree.Element("test", hours="4.0")
        assert _get_attr_int(elem, "hours") == 4

    def test_missing_returns_default(self) -> None:
        """Returns default when attribute is absent."""
        elem = etree.Element("test")
        assert _get_attr_int(elem, "hours", 8) == 8

    def test_missing_returns_none_default(self) -> None:
        """Returns None when attribute is absent and default is None."""
        elem = etree.Element("test")
        assert _get_attr_int(elem, "hours") is None


class TestGetAttrStr:
    """Tests for the _get_attr_str helper function."""

    def test_present_value(self) -> None:
        """Returns string when attribute is present."""
        elem = etree.Element("test", status="active")
        assert _get_attr_str(elem, "status") == "active"

    def test_missing_returns_default(self) -> None:
        """Returns default when attribute is absent."""
        elem = etree.Element("test")
        assert _get_attr_str(elem, "status", "unknown") == "unknown"

    def test_missing_returns_empty_default(self) -> None:
        """Returns empty string when attribute is absent with default."""
        elem = etree.Element("test")
        assert _get_attr_str(elem, "status") == ""


# ======================================================================
# UCXMLLoader: initialization
# ======================================================================


class TestUCXMLLoaderInit:
    """Tests for UCXMLLoader construction."""

    def test_init_creates_empty_defaults(self) -> None:
        """Loader starts with an empty defaults dictionary."""
        loader = UCXMLLoader()
        assert loader.defaults == {}

    def test_loader_is_reusable(self) -> None:
        """Loader can be used to load from multiple XML files."""
        loader = UCXMLLoader()
        assert isinstance(loader, UCXMLLoader)


# ======================================================================
# UCXMLLoader: loading generators with ALL UC attributes
# ======================================================================


class TestLoadFullUCAttributes:
    """Tests for loading generators from XML with all UC attributes present."""

    def test_loads_all_uc_attributes(self, tmp_path: Path) -> None:
        """Generator with complete UC attributes loads all values from XML."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Test Coal Plant"
                       capacity_mw="500" fuel_type="coal" status="active"
                       connected_substation="sub_001"
                       startup_cost="8000" shutdown_cost="3000"
                       min_up_time_h="6" min_down_time_h="4"
                       fuel_cost_per_mwh="4500" labor_cost_per_h="1200"
                       no_load_cost="600">
              <Location xmlns="urn:japan-grid:v1"
                        latitude="33.8" longitude="133.5"/>
              <RampRates xmlns="urn:japan-grid:v1"
                         ramp_up_mw_per_h="100" ramp_down_mw_per_h="80"/>
              <MaintenancePlan xmlns="urn:japan-grid:v1">
                <Window xmlns="urn:japan-grid:v1"
                        start="2000-01-02T00:00:00" end="2000-01-03T00:00:00"/>
              </MaintenancePlan>
              <RebuildPlan xmlns="urn:japan-grid:v1" planned_date="2030-06-01"/>
              <DisasterRisk xmlns="urn:japan-grid:v1" risk_score="0.35"/>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        assert len(generators) == 1
        gen = generators[0]

        # Required attributes
        assert gen.id == "gen_001"
        assert gen.name == "Test Coal Plant"
        assert gen.capacity_mw == 500.0
        assert gen.fuel_type == "coal"
        assert gen.status == "active"
        assert gen.connected_bus_id == "sub_001"

        # Location
        assert gen.latitude == 33.8
        assert gen.longitude == 133.5

        # UC attributes from XML (not defaults)
        assert gen.startup_cost == 8000.0
        assert gen.shutdown_cost == 3000.0
        assert gen.min_up_time_h == 6
        assert gen.min_down_time_h == 4
        assert gen.fuel_cost_per_mwh == 4500.0
        assert gen.labor_cost_per_h == 1200.0
        assert gen.no_load_cost == 600.0

        # RampRates child element
        assert gen.ramp_up_mw_per_h == 100.0
        assert gen.ramp_down_mw_per_h == 80.0

        # MaintenancePlan child element (2000-01-02T00:00 = hour 24, 2000-01-03T00:00 = hour 48)
        assert gen.maintenance_windows == [(24, 48)]

        # RebuildPlan child element
        assert gen.rebuild_planned_date == "2030-06-01"

        # DisasterRisk child element
        assert gen.disaster_risk_score == 0.35

    def test_loads_multiple_generators(self, tmp_path: Path) -> None:
        """Multiple generators in one region are all loaded."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Plant A"
                       capacity_mw="500" fuel_type="coal" status="active"/>
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_002" name="Plant B"
                       capacity_mw="300" fuel_type="lng" status="active"/>
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_003" name="Plant C"
                       capacity_mw="100" fuel_type="hydro" status="active"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        assert len(generators) == 3
        ids = [g.id for g in generators]
        assert ids == ["gen_001", "gen_002", "gen_003"]

    def test_loads_multiple_maintenance_windows(self, tmp_path: Path) -> None:
        """Multiple maintenance windows are all parsed."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Plant A"
                       capacity_mw="500" fuel_type="coal" status="active">
              <MaintenancePlan xmlns="urn:japan-grid:v1">
                <Window xmlns="urn:japan-grid:v1"
                        start="2000-01-02T00:00:00" end="2000-01-03T00:00:00"/>
                <Window xmlns="urn:japan-grid:v1"
                        start="2000-01-05T00:00:00" end="2000-01-06T12:00:00"/>
              </MaintenancePlan>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        # Window 1: hour 24 to hour 48
        # Window 2: hour 96 to hour 132 (5 days * 24 = 120, + 12 = 132)
        assert len(gen.maintenance_windows) == 2
        assert gen.maintenance_windows[0] == (24, 48)
        assert gen.maintenance_windows[1] == (96, 132)

    def test_ramp_up_only(self, tmp_path: Path) -> None:
        """RampRates with only ramp_up_mw_per_h sets ramp_down to config default."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Plant A"
                       capacity_mw="500" fuel_type="coal" status="active">
              <RampRates xmlns="urn:japan-grid:v1" ramp_up_mw_per_h="150"/>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.ramp_up_mw_per_h == 150.0
        # Default ramp_down_mw_per_h is None (unlimited) from config
        assert gen.ramp_down_mw_per_h is None


# ======================================================================
# UCXMLLoader: loading from XML with basic attributes only
# ======================================================================


class TestLoadBasicAttributesOnly:
    """Tests for loading generators with no UC attributes (defaults applied)."""

    def test_basic_generator_gets_config_defaults(self, tmp_path: Path) -> None:
        """Generator with no UC attributes receives all defaults from config."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Basic Plant"
                       capacity_mw="200" fuel_type="coal" status="active"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        assert len(generators) == 1
        gen = generators[0]

        # Defaults from config
        assert gen.startup_cost == 5000.0
        assert gen.shutdown_cost == 2000.0
        assert gen.min_up_time_h == 4
        assert gen.min_down_time_h == 4
        assert gen.labor_cost_per_h == 1000.0
        assert gen.no_load_cost == 500.0
        assert gen.disaster_risk_score == 0.0

        # Fuel-type-specific default for coal
        assert gen.fuel_cost_per_mwh == 4500.0

        # Ramp rates default to None (unlimited)
        assert gen.ramp_up_mw_per_h is None
        assert gen.ramp_down_mw_per_h is None

        # No maintenance, no rebuild, no location
        assert gen.maintenance_windows == []
        assert gen.rebuild_planned_date is None

    def test_fuel_type_specific_defaults(self, tmp_path: Path) -> None:
        """Different fuel types receive their specific default fuel costs."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_coal" name="Coal"
                       capacity_mw="500" fuel_type="coal" status="active"/>
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_lng" name="LNG"
                       capacity_mw="400" fuel_type="lng" status="active"/>
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_nuclear" name="Nuclear"
                       capacity_mw="1000" fuel_type="nuclear" status="active"/>
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_solar" name="Solar"
                       capacity_mw="50" fuel_type="solar" status="active"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen_map = {g.id: g for g in generators}
        assert gen_map["gen_coal"].fuel_cost_per_mwh == 4500.0
        assert gen_map["gen_lng"].fuel_cost_per_mwh == 7000.0
        assert gen_map["gen_nuclear"].fuel_cost_per_mwh == 1500.0
        assert gen_map["gen_solar"].fuel_cost_per_mwh == 0.0

    def test_unknown_fuel_type_uses_unknown_default(self, tmp_path: Path) -> None:
        """Generator with unrecognized fuel type gets 'unknown' default cost."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Mystery Plant"
                       capacity_mw="100" fuel_type="unknown" status="active"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.fuel_cost_per_mwh == 5000.0

    def test_no_location_defaults_to_zero(self, tmp_path: Path) -> None:
        """Generator without Location element defaults to (0.0, 0.0)."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="No Location"
                       capacity_mw="100" fuel_type="coal" status="active"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.latitude == 0.0
        assert gen.longitude == 0.0


# ======================================================================
# UCXMLLoader: loading from XML with partial UC attributes
# ======================================================================


class TestLoadPartialUCAttributes:
    """Tests for loading generators with some but not all UC attributes."""

    def test_partial_uc_attrs_mixed_with_defaults(self, tmp_path: Path) -> None:
        """Generator with some UC attrs gets remaining from config defaults."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Partial Plant"
                       capacity_mw="300" fuel_type="lng" status="active"
                       startup_cost="10000" min_up_time_h="8"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]

        # From XML
        assert gen.startup_cost == 10000.0
        assert gen.min_up_time_h == 8

        # From config defaults
        assert gen.shutdown_cost == 2000.0
        assert gen.min_down_time_h == 4
        assert gen.fuel_cost_per_mwh == 7000.0  # LNG default
        assert gen.labor_cost_per_h == 1000.0
        assert gen.no_load_cost == 500.0

    def test_ramp_rates_without_uc_attrs(self, tmp_path: Path) -> None:
        """Generator with RampRates but no UC attribute overrides."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Ramp Only Plant"
                       capacity_mw="300" fuel_type="coal" status="active">
              <RampRates xmlns="urn:japan-grid:v1"
                         ramp_up_mw_per_h="50" ramp_down_mw_per_h="40"/>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.ramp_up_mw_per_h == 50.0
        assert gen.ramp_down_mw_per_h == 40.0
        # Other UC attrs from defaults
        assert gen.startup_cost == 5000.0

    def test_location_with_no_uc_attrs(self, tmp_path: Path) -> None:
        """Generator with Location but no UC attrs gets location + defaults."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Located Plant"
                       capacity_mw="200" fuel_type="hydro" status="active">
              <Location xmlns="urn:japan-grid:v1"
                        latitude="35.6762" longitude="139.6503"/>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.latitude == 35.6762
        assert gen.longitude == 139.6503
        assert gen.fuel_cost_per_mwh == 0.0  # Hydro default

    def test_disaster_risk_only(self, tmp_path: Path) -> None:
        """Generator with only DisasterRisk child element."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Risky Plant"
                       capacity_mw="200" fuel_type="nuclear" status="active">
              <DisasterRisk xmlns="urn:japan-grid:v1" risk_score="0.75"/>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.disaster_risk_score == 0.75
        assert gen.fuel_cost_per_mwh == 1500.0  # Nuclear default

    def test_rebuild_plan_only(self, tmp_path: Path) -> None:
        """Generator with only RebuildPlan child element."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Aging Plant"
                       capacity_mw="300" fuel_type="coal" status="active">
              <RebuildPlan xmlns="urn:japan-grid:v1" planned_date="2028-12-15"/>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.rebuild_planned_date == "2028-12-15"

    def test_fuel_cost_from_xml_overrides_default(self, tmp_path: Path) -> None:
        """Explicit fuel_cost_per_mwh in XML overrides fuel-type default."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Cheap Coal"
                       capacity_mw="500" fuel_type="coal" status="active"
                       fuel_cost_per_mwh="2000"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        # XML value overrides config default (4500 for coal)
        assert gen.fuel_cost_per_mwh == 2000.0


# ======================================================================
# UCXMLLoader: error handling
# ======================================================================


class TestErrorHandling:
    """Tests for error handling in UCXMLLoader."""

    def test_malformed_xml_raises_syntax_error(self, tmp_path: Path) -> None:
        """Malformed XML raises lxml XMLSyntaxError."""
        xml_file = tmp_path / "bad.xml"
        xml_file.write_text("<not-valid><unclosed", encoding="utf-8")
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        with pytest.raises(etree.XMLSyntaxError):
            loader.load_generators_from_xml(str(xml_file), config_path)

    def test_missing_xml_file_raises_error(self, tmp_path: Path) -> None:
        """Non-existent XML file raises an appropriate error."""
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        with pytest.raises(OSError):
            loader.load_generators_from_xml(
                str(tmp_path / "nonexistent.xml"), config_path
            )

    def test_missing_config_file_raises_error(self, tmp_path: Path) -> None:
        """Non-existent config file raises FileNotFoundError."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Plant A"
                       capacity_mw="200" fuel_type="coal" status="active"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))

        loader = UCXMLLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_generators_from_xml(
                xml_path, str(tmp_path / "nonexistent_config.yaml")
            )

    def test_empty_generators_section(self, tmp_path: Path) -> None:
        """XML with empty Generators section returns empty list."""
        xml_content = _build_xml("")
        xml_path = _write_xml(tmp_path, xml_content)
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        assert generators == []

    def test_maintenance_window_missing_start(self, tmp_path: Path) -> None:
        """Maintenance window with missing start attribute is skipped."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Bad Maint"
                       capacity_mw="200" fuel_type="coal" status="active">
              <MaintenancePlan xmlns="urn:japan-grid:v1">
                <Window xmlns="urn:japan-grid:v1"
                        end="2000-01-03T00:00:00"/>
              </MaintenancePlan>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.maintenance_windows == []

    def test_maintenance_window_missing_end(self, tmp_path: Path) -> None:
        """Maintenance window with missing end attribute is skipped."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Bad Maint"
                       capacity_mw="200" fuel_type="coal" status="active">
              <MaintenancePlan xmlns="urn:japan-grid:v1">
                <Window xmlns="urn:japan-grid:v1"
                        start="2000-01-02T00:00:00"/>
              </MaintenancePlan>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.maintenance_windows == []

    def test_maintenance_window_invalid_datetime(self, tmp_path: Path) -> None:
        """Maintenance window with invalid datetime strings is skipped."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Bad DateTime"
                       capacity_mw="200" fuel_type="coal" status="active">
              <MaintenancePlan xmlns="urn:japan-grid:v1">
                <Window xmlns="urn:japan-grid:v1"
                        start="not-a-date" end="also-not-a-date"/>
              </MaintenancePlan>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.maintenance_windows == []

    def test_maintenance_window_start_after_end_skipped(self, tmp_path: Path) -> None:
        """Maintenance window where start >= end is skipped."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Bad Window"
                       capacity_mw="200" fuel_type="coal" status="active">
              <MaintenancePlan xmlns="urn:japan-grid:v1">
                <Window xmlns="urn:japan-grid:v1"
                        start="2000-01-03T00:00:00" end="2000-01-02T00:00:00"/>
              </MaintenancePlan>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.maintenance_windows == []

    def test_rebuild_plan_empty_date_returns_none(self, tmp_path: Path) -> None:
        """RebuildPlan with empty planned_date returns None."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Empty Rebuild"
                       capacity_mw="200" fuel_type="coal" status="active">
              <RebuildPlan xmlns="urn:japan-grid:v1" planned_date=""/>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.rebuild_planned_date is None


# ======================================================================
# UCXMLLoader: config defaults application
# ======================================================================


class TestConfigDefaults:
    """Tests for configuration defaults application in UCXMLLoader."""

    def test_custom_config_defaults_applied(self, tmp_path: Path) -> None:
        """Custom config defaults override the standard default values."""
        custom_config = {
            "defaults": {
                "startup_cost": 9999,
                "shutdown_cost": 8888,
                "min_up_time_h": 10,
                "min_down_time_h": 8,
                "ramp_up_mw_per_h": None,
                "ramp_down_mw_per_h": None,
                "fuel_cost_per_mwh": {"coal": 1234, "unknown": 5678},
                "labor_cost_per_h": 7777,
                "no_load_cost": 6666,
                "disaster_risk_score": 0.5,
            }
        }
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Custom Config"
                       capacity_mw="200" fuel_type="coal" status="active"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path, custom_config)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.startup_cost == 9999.0
        assert gen.shutdown_cost == 8888.0
        assert gen.min_up_time_h == 10
        assert gen.min_down_time_h == 8
        assert gen.fuel_cost_per_mwh == 1234.0
        assert gen.labor_cost_per_h == 7777.0
        assert gen.no_load_cost == 6666.0
        assert gen.disaster_risk_score == 0.5

    def test_config_without_defaults_section(self, tmp_path: Path) -> None:
        """Config file without defaults section uses fallback zero values."""
        empty_config = {"solver": {"backend": "cbc"}}
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="No Defaults Config"
                       capacity_mw="200" fuel_type="coal" status="active"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path, empty_config)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        # Falls back to hardcoded defaults (0.0 for float, 1 for int)
        assert gen.startup_cost == 0.0
        assert gen.shutdown_cost == 0.0
        assert gen.min_up_time_h == 1
        assert gen.min_down_time_h == 1
        assert gen.fuel_cost_per_mwh == 0.0
        assert gen.labor_cost_per_h == 0.0
        assert gen.no_load_cost == 0.0

    def test_loader_reloads_config_each_call(self, tmp_path: Path) -> None:
        """Each call to load_generators_from_xml reloads the config file."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Reload Test"
                       capacity_mw="200" fuel_type="coal" status="active"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))

        # First config
        config1 = {
            "defaults": {
                "startup_cost": 1000,
                "fuel_cost_per_mwh": {"coal": 100, "unknown": 100},
            }
        }
        config_path = _write_config(tmp_path, config1)
        loader = UCXMLLoader()
        gen1 = loader.load_generators_from_xml(xml_path, config_path)[0]
        assert gen1.startup_cost == 1000.0

        # Update config
        config2 = {
            "defaults": {
                "startup_cost": 9000,
                "fuel_cost_per_mwh": {"coal": 900, "unknown": 900},
            }
        }
        config_path2 = _write_config(tmp_path, config2)
        gen2 = loader.load_generators_from_xml(xml_path, config_path2)[0]
        assert gen2.startup_cost == 9000.0


# ======================================================================
# UCXMLLoader: multi-region loading
# ======================================================================


class TestMultiRegionLoading:
    """Tests for loading generators across multiple regions."""

    def test_loads_from_multiple_regions(self, tmp_path: Path) -> None:
        """Generators from multiple Region elements are all loaded."""
        xml_content = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <PowerGrid xmlns="urn:japan-grid:v1" version="1.0" created="2026-01-01">
              <Metadata>
                <Name>Test Grid</Name>
                <Description>Test</Description>
                <Source>Test</Source>
                <Attribution>Test</Attribution>
                <RegionCount>2</RegionCount>
              </Metadata>
              <Regions>
                <Region id="hokkaido" name="Hokkaido" frequency_hz="50">
                  <Substations/>
                  <TransmissionLines/>
                  <Generators>
                    <Generator id="hk_gen_001" name="Hokkaido Plant"
                               capacity_mw="500" fuel_type="coal" status="active"/>
                  </Generators>
                </Region>
                <Region id="shikoku" name="Shikoku" frequency_hz="60">
                  <Substations/>
                  <TransmissionLines/>
                  <Generators>
                    <Generator id="sk_gen_001" name="Shikoku Plant"
                               capacity_mw="300" fuel_type="lng" status="active"/>
                    <Generator id="sk_gen_002" name="Shikoku Plant 2"
                               capacity_mw="100" fuel_type="hydro" status="active"/>
                  </Generators>
                </Region>
              </Regions>
            </PowerGrid>
        """)
        xml_path = _write_xml(tmp_path, xml_content)
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        assert len(generators) == 3
        ids = [g.id for g in generators]
        assert "hk_gen_001" in ids
        assert "sk_gen_001" in ids
        assert "sk_gen_002" in ids

    def test_region_id_set_on_generators(self, tmp_path: Path) -> None:
        """Generators are assigned the correct region id from their parent Region."""
        xml_content = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <PowerGrid xmlns="urn:japan-grid:v1" version="1.0" created="2026-01-01">
              <Metadata>
                <Name>Test Grid</Name>
                <Description>Test</Description>
                <Source>Test</Source>
                <Attribution>Test</Attribution>
                <RegionCount>2</RegionCount>
              </Metadata>
              <Regions>
                <Region id="hokkaido" name="Hokkaido" frequency_hz="50">
                  <Substations/>
                  <TransmissionLines/>
                  <Generators>
                    <Generator id="hk_gen_001" name="Hokkaido Plant"
                               capacity_mw="500" fuel_type="coal" status="active"/>
                  </Generators>
                </Region>
                <Region id="shikoku" name="Shikoku" frequency_hz="60">
                  <Substations/>
                  <TransmissionLines/>
                  <Generators>
                    <Generator id="sk_gen_001" name="Shikoku Plant"
                               capacity_mw="300" fuel_type="lng" status="active"/>
                  </Generators>
                </Region>
              </Regions>
            </PowerGrid>
        """)
        xml_path = _write_xml(tmp_path, xml_content)
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen_map = {g.id: g for g in generators}
        assert gen_map["hk_gen_001"].region == "hokkaido"
        assert gen_map["sk_gen_001"].region == "shikoku"

    def test_region_without_generators_section(self, tmp_path: Path) -> None:
        """Region without a Generators element is skipped gracefully."""
        xml_content = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <PowerGrid xmlns="urn:japan-grid:v1" version="1.0" created="2026-01-01">
              <Metadata>
                <Name>Test Grid</Name>
                <Description>Test</Description>
                <Source>Test</Source>
                <Attribution>Test</Attribution>
                <RegionCount>2</RegionCount>
              </Metadata>
              <Regions>
                <Region id="hokkaido" name="Hokkaido" frequency_hz="50">
                  <Substations/>
                  <TransmissionLines/>
                </Region>
                <Region id="shikoku" name="Shikoku" frequency_hz="60">
                  <Substations/>
                  <TransmissionLines/>
                  <Generators>
                    <Generator id="sk_gen_001" name="Shikoku Plant"
                               capacity_mw="300" fuel_type="lng" status="active"/>
                  </Generators>
                </Region>
              </Regions>
            </PowerGrid>
        """)
        xml_path = _write_xml(tmp_path, xml_content)
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        assert len(generators) == 1
        assert generators[0].id == "sk_gen_001"


# ======================================================================
# UCXMLLoader: loading from actual sample_output.xml
# ======================================================================


class TestLoadSampleOutput:
    """Tests for loading generators from the project's sample_output.xml."""

    def test_loads_from_sample_output(
        self, project_root: Path, config_dir: Path
    ) -> None:
        """UCXMLLoader can load generators from the real sample_output.xml."""
        sample_xml = project_root / "schemas" / "sample_output.xml"
        config_yaml = config_dir / "uc_config.yaml"

        if not sample_xml.exists() or not config_yaml.exists():
            pytest.skip("sample_output.xml or uc_config.yaml not found")

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(
            str(sample_xml), str(config_yaml)
        )

        # sample_output.xml has 5 generators (3 Hokkaido + 2 Shikoku)
        assert len(generators) == 5

        # Verify first generator (Hokkaido coal plant)
        gen_map = {g.id: g for g in generators}
        hk_gen = gen_map["hokkaido_gen_001"]
        assert hk_gen.name == "苫東厚真発電所"
        assert hk_gen.capacity_mw == 1650.0
        assert hk_gen.fuel_type == "coal"
        assert hk_gen.region == "hokkaido"

        # Since sample XML has no UC attrs, all should be config defaults
        assert hk_gen.startup_cost == 5000.0  # config default
        assert hk_gen.fuel_cost_per_mwh == 4500.0  # coal default

    def test_sample_generators_have_locations(
        self, project_root: Path, config_dir: Path
    ) -> None:
        """Sample output generators have non-zero locations."""
        sample_xml = project_root / "schemas" / "sample_output.xml"
        config_yaml = config_dir / "uc_config.yaml"

        if not sample_xml.exists() or not config_yaml.exists():
            pytest.skip("sample_output.xml or uc_config.yaml not found")

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(
            str(sample_xml), str(config_yaml)
        )

        for gen in generators:
            assert gen.has_location, f"Generator {gen.id} should have location"


# ======================================================================
# UCXMLLoader: edge cases
# ======================================================================


class TestEdgeCases:
    """Tests for edge cases in UCXMLLoader."""

    def test_generator_with_zero_capacity(self, tmp_path: Path) -> None:
        """Generator with zero capacity loads without error."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Zero Cap"
                       capacity_mw="0" fuel_type="solar" status="planned"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        assert len(generators) == 1
        assert generators[0].capacity_mw == 0.0

    def test_generator_with_connected_substation(self, tmp_path: Path) -> None:
        """connected_substation attribute maps to connected_bus_id."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Connected"
                       capacity_mw="200" fuel_type="coal" status="active"
                       connected_substation="sub_abc_123"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        assert generators[0].connected_bus_id == "sub_abc_123"

    def test_generator_without_connected_substation(self, tmp_path: Path) -> None:
        """Generator without connected_substation has empty connected_bus_id."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Unconnected"
                       capacity_mw="200" fuel_type="solar" status="active"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        assert generators[0].connected_bus_id == ""

    def test_generator_status_defaults_to_active(self, tmp_path: Path) -> None:
        """Generator without explicit status defaults to 'active'."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="No Status"
                       capacity_mw="200" fuel_type="coal"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        assert generators[0].status == "active"

    def test_valid_maintenance_windows_kept_invalid_skipped(
        self, tmp_path: Path
    ) -> None:
        """Mix of valid and invalid maintenance windows: only valid kept."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Mixed Windows"
                       capacity_mw="200" fuel_type="coal" status="active">
              <MaintenancePlan xmlns="urn:japan-grid:v1">
                <Window xmlns="urn:japan-grid:v1"
                        start="2000-01-02T00:00:00" end="2000-01-03T00:00:00"/>
                <Window xmlns="urn:japan-grid:v1"
                        start="bad-date" end="bad-date"/>
                <Window xmlns="urn:japan-grid:v1"
                        start="2000-01-05T00:00:00" end="2000-01-06T00:00:00"/>
              </MaintenancePlan>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        # Only the two valid windows should be present
        assert len(gen.maintenance_windows) == 2
        assert gen.maintenance_windows[0] == (24, 48)
        assert gen.maintenance_windows[1] == (96, 120)

    def test_disaster_risk_without_score_uses_default(
        self, tmp_path: Path
    ) -> None:
        """DisasterRisk element without risk_score attribute uses config default."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="No Risk Score"
                       capacity_mw="200" fuel_type="coal" status="active">
              <DisasterRisk xmlns="urn:japan-grid:v1"/>
            </Generator>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        gen = generators[0]
        assert gen.disaster_risk_score == 0.0  # Config default

    def test_suspended_status_preserved(self, tmp_path: Path) -> None:
        """Generator with 'suspended' status preserves the status value."""
        gen_xml = textwrap.dedent("""\
            <Generator xmlns="urn:japan-grid:v1"
                       id="gen_001" name="Suspended Plant"
                       capacity_mw="200" fuel_type="nuclear" status="suspended"/>
        """)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        assert generators[0].status == "suspended"

    def test_all_fuel_types_load(self, tmp_path: Path) -> None:
        """Generators with all known fuel types load successfully."""
        fuel_types = [
            "coal", "lng", "oil", "nuclear", "hydro",
            "pumped_hydro", "geothermal", "wind", "solar", "biomass",
        ]
        gen_xml_parts = []
        for i, ft in enumerate(fuel_types, start=1):
            gen_xml_parts.append(
                f'<Generator xmlns="urn:japan-grid:v1"'
                f' id="gen_{i:03d}" name="Plant {ft}"'
                f' capacity_mw="100" fuel_type="{ft}" status="active"/>'
            )
        gen_xml = "\n".join(gen_xml_parts)
        xml_path = _write_xml(tmp_path, _build_xml(gen_xml))
        config_path = _write_config(tmp_path)

        loader = UCXMLLoader()
        generators = loader.load_generators_from_xml(xml_path, config_path)

        assert len(generators) == len(fuel_types)
        for gen, expected_ft in zip(generators, fuel_types):
            assert gen.fuel_type == expected_ft
