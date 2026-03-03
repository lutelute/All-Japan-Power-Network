"""Unit tests for the PandapowerBuilder module.

Tests pandapower network construction from GridNetwork data, including
bus creation, line creation with electrical parameters, generator creation,
external grid (slack bus) selection, frequency resolution, and edge cases
such as empty networks and missing bus references.
"""

import pandapower as pp
import pytest

from src.converter.pandapower_builder import (
    BuildResult,
    PandapowerBuilder,
    _EAST_50HZ_REGIONS,
    _WEST_60HZ_REGIONS,
)
from src.model.grid_network import GridNetwork
from src.model.substation import BusType

from tests.conftest import make_generator, make_substation, make_transmission_line


# ======================================================================
# BuildResult dataclass
# ======================================================================


class TestBuildResult:
    """Tests for the BuildResult dataclass."""

    def test_summary_keys(self) -> None:
        """Summary dict contains expected keys."""
        net = pp.create_empty_network()
        result = BuildResult(net=net, buses_created=3, lines_created=2)
        summary = result.summary
        assert "buses" in summary
        assert "lines" in summary
        assert "generators" in summary
        assert "ext_grids" in summary
        assert "warnings" in summary

    def test_default_values(self) -> None:
        """Default values are correct."""
        net = pp.create_empty_network()
        result = BuildResult(net=net)
        assert result.buses_created == 0
        assert result.lines_created == 0
        assert result.generators_created == 0
        assert result.ext_grids_created == 0
        assert result.warnings == []
        assert result.bus_map == {}


# ======================================================================
# PandapowerBuilder: initialization
# ======================================================================


class TestPandapowerBuilderInit:
    """Tests for PandapowerBuilder construction."""

    def test_default_frequency(self) -> None:
        """Default national frequency is 50 Hz."""
        builder = PandapowerBuilder()
        assert builder._default_national_f_hz == 50

    def test_custom_frequency(self) -> None:
        """Custom national frequency is accepted."""
        builder = PandapowerBuilder(default_national_f_hz=60)
        assert builder._default_national_f_hz == 60

    def test_invalid_frequency_raises(self) -> None:
        """Non-50/60 frequency raises ValueError."""
        with pytest.raises(ValueError, match="must be 50 or 60"):
            PandapowerBuilder(default_national_f_hz=45)


# ======================================================================
# PandapowerBuilder: build - bus creation
# ======================================================================


class TestBusBuild:
    """Tests for bus (substation) creation."""

    def test_bus_count(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """Bus count matches substation count in the network."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        assert result.buses_created == sample_grid_network.substation_count
        assert len(result.net.bus) == sample_grid_network.substation_count

    def test_bus_map_populated(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """bus_map contains all substation IDs."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        for sub in sample_grid_network.substations:
            assert sub.id in result.bus_map

    def test_bus_voltage(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """Bus nominal voltage matches substation voltage_kv."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        net = result.net

        for sub in sample_grid_network.substations:
            bus_idx = result.bus_map[sub.id]
            assert net.bus.at[bus_idx, "vn_kv"] == sub.voltage_kv

    def test_bus_name(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """Bus name matches substation name."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        net = result.net

        for sub in sample_grid_network.substations:
            bus_idx = result.bus_map[sub.id]
            assert net.bus.at[bus_idx, "name"] == sub.name

    def test_bus_zone_set(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """Bus zone column is set to the region."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        net = result.net

        for sub in sample_grid_network.substations:
            bus_idx = result.bus_map[sub.id]
            assert net.bus.at[bus_idx, "zone"] == sub.region

    def test_empty_network_raises(self) -> None:
        """Network with no substations raises ValueError."""
        builder = PandapowerBuilder()
        network = GridNetwork(region="test", frequency_hz=60)
        with pytest.raises(ValueError, match="no substations"):
            builder.build(network)


# ======================================================================
# PandapowerBuilder: build - line creation
# ======================================================================


class TestLineBuild:
    """Tests for line (transmission line) creation."""

    def test_line_count(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """Line count matches transmission line count."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        assert result.lines_created == sample_grid_network.line_count
        assert len(result.net.line) == sample_grid_network.line_count

    def test_line_parameters(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """Line electrical parameters are set correctly."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        net = result.net

        # First line in the fixture has known parameters
        first_line = sample_grid_network.lines[0]
        assert len(net.line) > 0

        line_row = net.line.iloc[0]
        assert line_row["r_ohm_per_km"] == first_line.r_ohm_per_km
        assert line_row["x_ohm_per_km"] == first_line.x_ohm_per_km
        assert line_row["c_nf_per_km"] == first_line.c_nf_per_km
        assert line_row["max_i_ka"] == first_line.max_i_ka
        assert line_row["length_km"] == first_line.length_km

    def test_line_from_to_buses(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """Lines reference correct from/to bus indices."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        net = result.net

        for i, line in enumerate(sample_grid_network.lines):
            expected_from = result.bus_map[line.from_substation_id]
            expected_to = result.bus_map[line.to_substation_id]
            assert net.line.at[i, "from_bus"] == expected_from
            assert net.line.at[i, "to_bus"] == expected_to

    def test_missing_from_bus_skipped(self) -> None:
        """Line with missing from_substation is skipped with warning."""
        network = GridNetwork(region="test", frequency_hz=60)
        sub_a = make_substation(id="sub_a")
        network.add_substation(sub_a)

        line = make_transmission_line(
            id="line_1",
            from_substation_id="nonexistent",
            to_substation_id="sub_a",
        )
        network.add_transmission_line(line)

        builder = PandapowerBuilder()
        result = builder.build(network)
        assert result.lines_created == 0
        assert len(result.warnings) > 0
        assert "nonexistent" in result.warnings[0]

    def test_missing_to_bus_skipped(self) -> None:
        """Line with missing to_substation is skipped with warning."""
        network = GridNetwork(region="test", frequency_hz=60)
        sub_a = make_substation(id="sub_a")
        network.add_substation(sub_a)

        line = make_transmission_line(
            id="line_1",
            from_substation_id="sub_a",
            to_substation_id="nonexistent",
        )
        network.add_transmission_line(line)

        builder = PandapowerBuilder()
        result = builder.build(network)
        assert result.lines_created == 0
        assert len(result.warnings) > 0

    def test_zero_length_line_skipped(self) -> None:
        """Zero-length line is skipped with warning."""
        network = GridNetwork(region="test", frequency_hz=60)
        sub_a = make_substation(id="sub_a")
        sub_b = make_substation(id="sub_b", name="SubB")
        network.add_substation(sub_a)
        network.add_substation(sub_b)

        line = make_transmission_line(
            id="line_1",
            from_substation_id="sub_a",
            to_substation_id="sub_b",
            length_km=0.0,
        )
        network.add_transmission_line(line)

        builder = PandapowerBuilder()
        result = builder.build(network)
        assert result.lines_created == 0
        assert any("zero-length" in w for w in result.warnings)


# ======================================================================
# PandapowerBuilder: build - generator creation
# ======================================================================


class TestGeneratorBuild:
    """Tests for generator creation."""

    def test_generator_count(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """Generator count matches connected generators."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        assert result.generators_created == sample_grid_network.generator_count
        assert len(result.net.gen) == sample_grid_network.generator_count

    def test_generator_capacity(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """Generator p_mw matches capacity_mw."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        net = result.net

        first_gen = sample_grid_network.generators[0]
        assert net.gen.iloc[0]["p_mw"] == first_gen.capacity_mw

    def test_generator_vm_pu(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """Generator vm_pu setpoint is preserved."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        net = result.net

        first_gen = sample_grid_network.generators[0]
        assert net.gen.iloc[0]["vm_pu"] == first_gen.vm_pu

    def test_unconnected_generator_skipped(self) -> None:
        """Generator without connected_bus_id is skipped."""
        network = GridNetwork(region="test", frequency_hz=60)
        sub = make_substation(id="sub_a")
        network.add_substation(sub)

        gen = make_generator(
            id="gen_1",
            connected_bus_id="",
        )
        network.add_generator(gen)

        builder = PandapowerBuilder()
        result = builder.build(network)
        assert result.generators_created == 0
        assert any("not connected" in w for w in result.warnings)

    def test_generator_with_missing_bus_skipped(self) -> None:
        """Generator referencing nonexistent bus is skipped."""
        network = GridNetwork(region="test", frequency_hz=60)
        sub = make_substation(id="sub_a")
        network.add_substation(sub)

        gen = make_generator(
            id="gen_1",
            connected_bus_id="nonexistent_bus",
        )
        network.add_generator(gen)

        builder = PandapowerBuilder()
        result = builder.build(network)
        assert result.generators_created == 0
        assert any("not found" in w for w in result.warnings)


# ======================================================================
# PandapowerBuilder: build - ext_grid (slack bus)
# ======================================================================


class TestExtGridBuild:
    """Tests for external grid (slack bus) creation."""

    def test_ext_grid_created(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """At least one ext_grid is created for every network."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        assert result.ext_grids_created == 1
        assert len(result.net.ext_grid) == 1

    def test_slack_from_explicit_bus_type(self) -> None:
        """Explicit SLACK bus type is selected for ext_grid."""
        network = GridNetwork(region="test", frequency_hz=60)
        sub_pq = make_substation(
            id="sub_pq", name="PQ Bus",
            bus_type=BusType.PQ.value,
        )
        sub_slack = make_substation(
            id="sub_slack", name="Slack Bus",
            bus_type=BusType.SLACK.value,
        )
        network.add_substation(sub_pq)
        network.add_substation(sub_slack)

        builder = PandapowerBuilder()
        result = builder.build(network)
        slack_bus_idx = result.bus_map["sub_slack"]
        assert result.net.ext_grid.iloc[0]["bus"] == slack_bus_idx

    def test_slack_from_largest_generator(self) -> None:
        """Without explicit SLACK, the largest generator's bus is selected."""
        network = GridNetwork(region="test", frequency_hz=60)
        sub_a = make_substation(id="sub_a", name="SubA")
        sub_b = make_substation(id="sub_b", name="SubB")
        network.add_substation(sub_a)
        network.add_substation(sub_b)

        gen_small = make_generator(
            id="gen_small", capacity_mw=100.0,
            connected_bus_id="sub_a",
        )
        gen_large = make_generator(
            id="gen_large", name="Large Gen",
            capacity_mw=1000.0,
            connected_bus_id="sub_b",
        )
        network.add_generator(gen_small)
        network.add_generator(gen_large)

        builder = PandapowerBuilder()
        result = builder.build(network)
        # Slack should be at sub_b (largest gen)
        slack_bus_idx = result.bus_map["sub_b"]
        assert result.net.ext_grid.iloc[0]["bus"] == slack_bus_idx

    def test_slack_fallback_first_bus(self) -> None:
        """Without generators or explicit SLACK, first bus is used."""
        network = GridNetwork(region="test", frequency_hz=60)
        sub_a = make_substation(id="sub_a", name="First Bus")
        sub_b = make_substation(id="sub_b", name="Second Bus")
        network.add_substation(sub_a)
        network.add_substation(sub_b)

        builder = PandapowerBuilder()
        result = builder.build(network)
        first_bus_idx = result.bus_map["sub_a"]
        assert result.net.ext_grid.iloc[0]["bus"] == first_bus_idx
        # Warning should be issued for fallback
        assert any("first bus" in w.lower() for w in result.warnings)


# ======================================================================
# PandapowerBuilder: frequency resolution
# ======================================================================


class TestFrequencyResolution:
    """Tests for network frequency resolution."""

    def test_60hz_region(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """Shikoku (60 Hz) network uses 60 Hz."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        assert result.net.f_hz == 60

    def test_50hz_region(
        self,
        sample_grid_network_50hz: GridNetwork,
    ) -> None:
        """Hokkaido (50 Hz) network uses 50 Hz."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network_50hz)
        assert result.net.f_hz == 50

    def test_mixed_frequency_uses_default(self) -> None:
        """Network with frequency_hz=0 uses the builder's default."""
        network = GridNetwork(region="national", frequency_hz=0)
        sub = make_substation(id="sub_a")
        network.add_substation(sub)

        builder = PandapowerBuilder(default_national_f_hz=50)
        result = builder.build(network)
        assert result.net.f_hz == 50

    def test_mixed_frequency_uses_custom_default(self) -> None:
        """Custom default frequency is applied for mixed models."""
        network = GridNetwork(region="national", frequency_hz=0)
        sub = make_substation(id="sub_a")
        network.add_substation(sub)

        builder = PandapowerBuilder(default_national_f_hz=60)
        result = builder.build(network)
        assert result.net.f_hz == 60


# ======================================================================
# PandapowerBuilder: get_region_frequency
# ======================================================================


class TestGetRegionFrequency:
    """Tests for the static region frequency lookup."""

    def test_east_regions_50hz(self) -> None:
        """East Japan regions return 50 Hz."""
        for region in _EAST_50HZ_REGIONS:
            assert PandapowerBuilder.get_region_frequency(region) == 50

    def test_west_regions_60hz(self) -> None:
        """West Japan regions return 60 Hz."""
        for region in _WEST_60HZ_REGIONS:
            assert PandapowerBuilder.get_region_frequency(region) == 60

    def test_unknown_region_raises(self) -> None:
        """Unknown region name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown region"):
            PandapowerBuilder.get_region_frequency("atlantis")


# ======================================================================
# PandapowerBuilder: network naming
# ======================================================================


class TestNetworkNaming:
    """Tests for pandapower network name."""

    def test_network_name(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """Network name includes region identifier."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)
        assert "shikoku" in result.net.name


# ======================================================================
# PandapowerBuilder: full build integration
# ======================================================================


class TestFullBuild:
    """Integration-style tests for the complete build process."""

    def test_complete_build(
        self,
        sample_grid_network: GridNetwork,
    ) -> None:
        """Complete build creates buses, lines, generators, and ext_grid."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network)

        assert result.buses_created == 5
        assert result.lines_created == 3
        assert result.generators_created == 3
        assert result.ext_grids_created == 1

    def test_50hz_build(
        self,
        sample_grid_network_50hz: GridNetwork,
    ) -> None:
        """50 Hz build produces correct element counts."""
        builder = PandapowerBuilder()
        result = builder.build(sample_grid_network_50hz)

        assert result.buses_created == 2
        assert result.lines_created == 1
        assert result.generators_created == 1
        assert result.ext_grids_created == 1
        assert result.net.f_hz == 50

    def test_network_without_lines(self) -> None:
        """Network with substations but no lines builds successfully."""
        network = GridNetwork(region="test", frequency_hz=60)
        sub = make_substation(id="sub_a")
        network.add_substation(sub)

        builder = PandapowerBuilder()
        result = builder.build(network)
        assert result.buses_created == 1
        assert result.lines_created == 0
        assert result.ext_grids_created == 1

    def test_network_without_generators(self) -> None:
        """Network with substations and lines but no generators builds."""
        network = GridNetwork(region="test", frequency_hz=60)
        sub_a = make_substation(id="sub_a")
        sub_b = make_substation(id="sub_b", name="SubB")
        network.add_substation(sub_a)
        network.add_substation(sub_b)

        line = make_transmission_line(
            id="line_1",
            from_substation_id="sub_a",
            to_substation_id="sub_b",
        )
        network.add_transmission_line(line)

        builder = PandapowerBuilder()
        result = builder.build(network)
        assert result.buses_created == 2
        assert result.lines_created == 1
        assert result.generators_created == 0
        assert result.ext_grids_created == 1
