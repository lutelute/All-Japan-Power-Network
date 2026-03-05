"""Shared pytest fixtures for JPGrid-Open test suite.

Provides reusable fixtures for:
- Mock Substation, TransmissionLine, and Generator objects
- Sample GridNetwork instances (regional and national)
- Temporary output directories for test isolation
"""

from pathlib import Path
from typing import List

import pytest

from src.model.generator import Generator
from src.model.grid_network import GridNetwork
from src.model.substation import (
    BusType,
    CapacityStatus,
    FuelType,
    Substation,
    VoltageClass,
)
from src.model.transmission_line import TransmissionLine
from src.uc.models import Interconnection


# ======================================================================
# Project paths
# ======================================================================


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Return the config directory path."""
    return project_root / "config"


@pytest.fixture
def schemas_dir(project_root: Path) -> Path:
    """Return the schemas directory path."""
    return project_root / "schemas"


@pytest.fixture
def xsd_path(schemas_dir: Path) -> Path:
    """Return the path to power_grid.xsd schema file."""
    return schemas_dir / "power_grid.xsd"


# ======================================================================
# Temporary output directories
# ======================================================================


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create and return a temporary output directory structure.

    Mirrors the project's output/ directory layout:
        output/xml/regions/
        output/matpower/regions/
        output/reports/
    """
    xml_dir = tmp_path / "output" / "xml" / "regions"
    xml_dir.mkdir(parents=True)

    matpower_dir = tmp_path / "output" / "matpower" / "regions"
    matpower_dir.mkdir(parents=True)

    reports_dir = tmp_path / "output" / "reports"
    reports_dir.mkdir(parents=True)

    return tmp_path / "output"


@pytest.fixture
def tmp_xml_dir(tmp_output_dir: Path) -> Path:
    """Return the temporary XML output directory."""
    return tmp_output_dir / "xml"


@pytest.fixture
def tmp_matpower_dir(tmp_output_dir: Path) -> Path:
    """Return the temporary MATPOWER output directory."""
    return tmp_output_dir / "matpower"


@pytest.fixture
def tmp_reports_dir(tmp_output_dir: Path) -> Path:
    """Return the temporary reports output directory."""
    return tmp_output_dir / "reports"



# ======================================================================
# Sample Substation fixtures
# ======================================================================


def make_substation(
    id: str = "shikoku_sub_001",
    name: str = "Test変電所",
    region: str = "shikoku",
    latitude: float = 33.8,
    longitude: float = 133.5,
    voltage_kv: float = 275.0,
    bus_type: int = BusType.PQ.value,
    voltage_class: VoltageClass = None,
    source_map: str = "shikoku.kml",
    grid_class: str = "backbone",
    description: str = "",
) -> Substation:
    """Factory function for creating Substation instances in tests.

    Provides sensible defaults for the Shikoku region while allowing
    any field to be overridden.
    """
    return Substation(
        id=id,
        name=name,
        region=region,
        latitude=latitude,
        longitude=longitude,
        voltage_kv=voltage_kv,
        bus_type=bus_type,
        voltage_class=voltage_class,
        source_map=source_map,
        grid_class=grid_class,
        description=description,
    )


@pytest.fixture
def substation_500kv() -> Substation:
    """Return a 500kV backbone substation."""
    return make_substation(
        id="shikoku_sub_001",
        name="阿南変電所",
        voltage_kv=500.0,
        latitude=33.9167,
        longitude=134.6500,
        grid_class="backbone",
    )


@pytest.fixture
def substation_275kv() -> Substation:
    """Return a 275kV backbone substation."""
    return make_substation(
        id="shikoku_sub_002",
        name="讃岐変電所",
        voltage_kv=275.0,
        latitude=34.2000,
        longitude=134.0000,
        grid_class="backbone",
    )


@pytest.fixture
def substation_regional() -> Substation:
    """Return a regional voltage substation (187kV for Shikoku)."""
    return make_substation(
        id="shikoku_sub_003",
        name="高松変電所",
        voltage_kv=187.0,
        latitude=34.3500,
        longitude=134.0500,
        grid_class="regional",
    )


@pytest.fixture
def substation_slack() -> Substation:
    """Return a substation designated as the slack bus."""
    return make_substation(
        id="shikoku_sub_004",
        name="本川変電所",
        voltage_kv=500.0,
        latitude=33.8000,
        longitude=133.2000,
        bus_type=BusType.SLACK.value,
        grid_class="backbone",
    )


@pytest.fixture
def substation_generator_bus() -> Substation:
    """Return a substation designated as a PV (generator) bus."""
    return make_substation(
        id="shikoku_sub_005",
        name="伊方変電所",
        voltage_kv=500.0,
        latitude=33.4900,
        longitude=132.3100,
        bus_type=BusType.PV.value,
        grid_class="backbone",
    )


@pytest.fixture
def sample_substations(
    substation_500kv: Substation,
    substation_275kv: Substation,
    substation_regional: Substation,
    substation_slack: Substation,
    substation_generator_bus: Substation,
) -> List[Substation]:
    """Return a list of 5 substations covering all referenced IDs.

    Includes sub_004 (slack) and sub_005 (generator bus) which are
    referenced by transmission lines and generators in the sample network.
    """
    return [
        substation_500kv,
        substation_275kv,
        substation_regional,
        substation_slack,
        substation_generator_bus,
    ]


# ======================================================================
# Sample TransmissionLine fixtures
# ======================================================================


def make_transmission_line(
    id: str = "shikoku_line_001",
    name: str = "Test送電線",
    from_substation_id: str = "shikoku_sub_001",
    to_substation_id: str = "shikoku_sub_002",
    voltage_kv: float = 275.0,
    length_km: float = 45.2,
    region: str = "shikoku",
    r_ohm_per_km: float = 0.028,
    x_ohm_per_km: float = 0.325,
    c_nf_per_km: float = 12.24,
    max_i_ka: float = 2.0,
    capacity_status: CapacityStatus = CapacityStatus.AVAILABLE,
    voltage_class: VoltageClass = None,
    n1_eligible: bool = False,
    grid_class: str = "backbone",
    coordinates: list = None,
    source_map: str = "shikoku.kml",
    description: str = "",
) -> TransmissionLine:
    """Factory function for creating TransmissionLine instances in tests.

    Provides sensible defaults for a 275kV backbone line while allowing
    any field to be overridden.
    """
    if coordinates is None:
        coordinates = [
            (33.9167, 134.6500),
            (34.0500, 134.3000),
            (34.2000, 134.0000),
        ]
    return TransmissionLine(
        id=id,
        name=name,
        from_substation_id=from_substation_id,
        to_substation_id=to_substation_id,
        voltage_kv=voltage_kv,
        length_km=length_km,
        region=region,
        r_ohm_per_km=r_ohm_per_km,
        x_ohm_per_km=x_ohm_per_km,
        c_nf_per_km=c_nf_per_km,
        max_i_ka=max_i_ka,
        capacity_status=capacity_status,
        voltage_class=voltage_class,
        n1_eligible=n1_eligible,
        grid_class=grid_class,
        coordinates=coordinates,
        source_map=source_map,
        description=description,
    )


@pytest.fixture
def line_500kv_red() -> TransmissionLine:
    """Return a 500kV line with zero capacity, N-1 ineligible (red)."""
    return make_transmission_line(
        id="shikoku_line_001",
        name="阿南幹線",
        from_substation_id="shikoku_sub_001",
        to_substation_id="shikoku_sub_004",
        voltage_kv=500.0,
        length_km=80.5,
        r_ohm_per_km=0.012,
        x_ohm_per_km=0.290,
        c_nf_per_km=13.03,
        max_i_ka=4.0,
        capacity_status=CapacityStatus.ZERO_N1_INELIGIBLE,
        grid_class="backbone",
    )


@pytest.fixture
def line_275kv_orange() -> TransmissionLine:
    """Return a 275kV line with zero capacity, N-1 eligible (orange)."""
    return make_transmission_line(
        id="shikoku_line_002",
        name="讃岐連絡線",
        from_substation_id="shikoku_sub_002",
        to_substation_id="shikoku_sub_003",
        voltage_kv=275.0,
        length_km=35.0,
        r_ohm_per_km=0.028,
        x_ohm_per_km=0.325,
        c_nf_per_km=12.24,
        max_i_ka=2.0,
        capacity_status=CapacityStatus.ZERO_N1_ELIGIBLE,
        grid_class="backbone",
    )


@pytest.fixture
def line_regional_blue() -> TransmissionLine:
    """Return a regional voltage line with available capacity (blue)."""
    return make_transmission_line(
        id="shikoku_line_003",
        name="高松配電線",
        from_substation_id="shikoku_sub_003",
        to_substation_id="shikoku_sub_001",
        voltage_kv=187.0,
        length_km=55.0,
        r_ohm_per_km=0.050,
        x_ohm_per_km=0.380,
        c_nf_per_km=9.28,
        max_i_ka=1.0,
        capacity_status=CapacityStatus.AVAILABLE,
        grid_class="regional",
    )


@pytest.fixture
def sample_lines(
    line_500kv_red: TransmissionLine,
    line_275kv_orange: TransmissionLine,
    line_regional_blue: TransmissionLine,
) -> List[TransmissionLine]:
    """Return a list of 3 transmission lines at different voltage/status."""
    return [line_500kv_red, line_275kv_orange, line_regional_blue]


# ======================================================================
# Sample Generator fixtures
# ======================================================================


def make_generator(
    id: str = "shikoku_gen_001",
    name: str = "Test発電所",
    capacity_mw: float = 500.0,
    fuel_type: str = "coal",
    connected_bus_id: str = "shikoku_sub_001",
    region: str = "shikoku",
    latitude: float = 33.8,
    longitude: float = 133.5,
    operator: str = "四国電力",
    status: str = "active",
    vm_pu: float = 1.0,
    p_min_mw: float = 0.0,
    source: str = "P03",
    description: str = "",
    # UC-specific fields
    startup_cost: float = 0.0,
    shutdown_cost: float = 0.0,
    min_up_time_h: int = 1,
    min_down_time_h: int = 1,
    ramp_up_mw_per_h: float = None,
    ramp_down_mw_per_h: float = None,
    fuel_cost_per_mwh: float = 0.0,
    labor_cost_per_h: float = 0.0,
    no_load_cost: float = 0.0,
    maintenance_windows: list = None,
    construction_date: str = None,
    rebuild_planned_date: str = None,
    disaster_risk_score: float = 0.0,
    # Storage fields
    storage_capacity_mwh: float = 0.0,
    charge_rate_mw: float = None,
    discharge_rate_mw: float = None,
    charge_efficiency: float = 0.90,
    discharge_efficiency: float = 0.90,
    initial_soc_fraction: float = 0.5,
    min_terminal_soc_fraction: float = 0.5,
) -> Generator:
    """Factory function for creating Generator instances in tests.

    Provides sensible defaults for a coal plant in Shikoku while
    allowing any field to be overridden. Supports base fields,
    UC-specific fields (startup/shutdown costs, ramp rates, etc.),
    and storage fields (capacity, charge/discharge rates, efficiencies).
    """
    if maintenance_windows is None:
        maintenance_windows = []
    return Generator(
        id=id,
        name=name,
        capacity_mw=capacity_mw,
        fuel_type=fuel_type,
        connected_bus_id=connected_bus_id,
        region=region,
        latitude=latitude,
        longitude=longitude,
        operator=operator,
        status=status,
        vm_pu=vm_pu,
        p_min_mw=p_min_mw,
        source=source,
        description=description,
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
        construction_date=construction_date,
        rebuild_planned_date=rebuild_planned_date,
        disaster_risk_score=disaster_risk_score,
        storage_capacity_mwh=storage_capacity_mwh,
        charge_rate_mw=charge_rate_mw,
        discharge_rate_mw=discharge_rate_mw,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
        initial_soc_fraction=initial_soc_fraction,
        min_terminal_soc_fraction=min_terminal_soc_fraction,
    )


def make_storage_generator(
    id: str = "storage_gen_001",
    name: str = "揚水発電所",
    capacity_mw: float = 100.0,
    fuel_type: str = "hydro",
    storage_capacity_mwh: float = 400.0,
    charge_rate_mw: float = 100.0,
    discharge_rate_mw: float = 100.0,
    charge_efficiency: float = 0.85,
    discharge_efficiency: float = 0.90,
    initial_soc_fraction: float = 0.5,
    min_terminal_soc_fraction: float = 0.5,
    **kwargs,
) -> Generator:
    """Convenience factory for creating pumped hydro storage generators.

    Provides sensible defaults for a pumped hydro unit: 100 MW capacity,
    400 MWh storage, 85% charge efficiency, 90% discharge efficiency,
    50% initial SOC. All keyword arguments are forwarded to
    ``make_generator()``.

    Args:
        id: Generator identifier.
        name: Generator name (default: 揚水発電所 = pumped hydro plant).
        capacity_mw: Rated power capacity in MW.
        fuel_type: Fuel type classification.
        storage_capacity_mwh: Energy storage capacity in MWh.
        charge_rate_mw: Maximum charging rate in MW.
        discharge_rate_mw: Maximum discharging rate in MW.
        charge_efficiency: Charging efficiency fraction (0, 1].
        discharge_efficiency: Discharging efficiency fraction (0, 1].
        initial_soc_fraction: Initial SOC as fraction of capacity.
        min_terminal_soc_fraction: Minimum terminal SOC fraction.
        **kwargs: Additional keyword arguments forwarded to make_generator.

    Returns:
        Generator instance configured as pumped hydro storage.
    """
    return make_generator(
        id=id,
        name=name,
        capacity_mw=capacity_mw,
        fuel_type=fuel_type,
        storage_capacity_mwh=storage_capacity_mwh,
        charge_rate_mw=charge_rate_mw,
        discharge_rate_mw=discharge_rate_mw,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
        initial_soc_fraction=initial_soc_fraction,
        min_terminal_soc_fraction=min_terminal_soc_fraction,
        **kwargs,
    )


# ======================================================================
# Interconnection factory
# ======================================================================


def make_interconnection(
    id: str = "ic_test_001",
    name_en: str = "Test Interconnection",
    from_region: str = "tokyo",
    to_region: str = "chubu",
    capacity_mw: float = 1000.0,
    type: str = "AC",
) -> Interconnection:
    """Factory function for creating Interconnection instances in tests.

    Provides sensible defaults for a Tokyo-Chubu AC interconnection
    while allowing any field to be overridden.
    """
    return Interconnection(
        id=id,
        name_en=name_en,
        from_region=from_region,
        to_region=to_region,
        capacity_mw=capacity_mw,
        type=type,
    )


@pytest.fixture
def generator_coal() -> Generator:
    """Return a coal-fired thermal generator."""
    return make_generator(
        id="shikoku_gen_001",
        name="坂出発電所",
        capacity_mw=1460.0,
        fuel_type="coal",
        connected_bus_id="shikoku_sub_002",
        latitude=34.3100,
        longitude=133.8500,
        operator="四国電力",
    )


@pytest.fixture
def generator_nuclear() -> Generator:
    """Return a nuclear generator."""
    return make_generator(
        id="shikoku_gen_002",
        name="伊方発電所",
        capacity_mw=890.0,
        fuel_type="nuclear",
        connected_bus_id="shikoku_sub_005",
        latitude=33.4900,
        longitude=132.3100,
        operator="四国電力",
    )


@pytest.fixture
def generator_hydro() -> Generator:
    """Return a hydroelectric generator."""
    return make_generator(
        id="shikoku_gen_003",
        name="早明浦発電所",
        capacity_mw=63.0,
        fuel_type="hydro",
        connected_bus_id="shikoku_sub_003",
        latitude=33.8200,
        longitude=133.4500,
        operator="四国電力",
    )


@pytest.fixture
def generator_unconnected() -> Generator:
    """Return a generator with no connected substation bus."""
    return make_generator(
        id="shikoku_gen_004",
        name="未接続発電所",
        capacity_mw=50.0,
        fuel_type="solar",
        connected_bus_id="",
        latitude=34.0000,
        longitude=133.8000,
    )


@pytest.fixture
def sample_generators(
    generator_coal: Generator,
    generator_nuclear: Generator,
    generator_hydro: Generator,
) -> List[Generator]:
    """Return a list of 3 generators with different fuel types."""
    return [generator_coal, generator_nuclear, generator_hydro]


# ======================================================================
# Sample GridNetwork fixtures
# ======================================================================


@pytest.fixture
def empty_grid_network() -> GridNetwork:
    """Return an empty GridNetwork for the Shikoku region."""
    return GridNetwork(region="shikoku", frequency_hz=60)


@pytest.fixture
def sample_grid_network(
    sample_substations: List[Substation],
    sample_lines: List[TransmissionLine],
    sample_generators: List[Generator],
) -> GridNetwork:
    """Return a populated GridNetwork for the Shikoku region.

    Contains 3 substations, 3 transmission lines, and 3 generators
    covering various voltage levels, capacity statuses, and fuel types.
    """
    network = GridNetwork(region="shikoku", frequency_hz=60)

    for sub in sample_substations:
        network.add_substation(sub)

    for line in sample_lines:
        network.add_transmission_line(line)

    for gen in sample_generators:
        network.add_generator(gen)

    return network


@pytest.fixture
def sample_grid_network_50hz() -> GridNetwork:
    """Return a populated GridNetwork for Hokkaido (50 Hz region).

    Minimal network with 2 substations and 1 line for frequency-related
    testing (e.g., B→c_nf_per_km conversion at 50 Hz).
    """
    network = GridNetwork(region="hokkaido", frequency_hz=50)

    sub_a = make_substation(
        id="hokkaido_sub_001",
        name="北海道変電所A",
        region="hokkaido",
        latitude=43.0621,
        longitude=141.3544,
        voltage_kv=275.0,
        source_map="hokkaido.kml",
    )
    sub_b = make_substation(
        id="hokkaido_sub_002",
        name="北海道変電所B",
        region="hokkaido",
        latitude=43.2000,
        longitude=141.5000,
        voltage_kv=275.0,
        source_map="hokkaido.kml",
    )
    network.add_substation(sub_a)
    network.add_substation(sub_b)

    line = make_transmission_line(
        id="hokkaido_line_001",
        name="北海道幹線",
        from_substation_id="hokkaido_sub_001",
        to_substation_id="hokkaido_sub_002",
        voltage_kv=275.0,
        length_km=20.0,
        region="hokkaido",
        capacity_status=CapacityStatus.AVAILABLE,
    )
    network.add_transmission_line(line)

    gen = make_generator(
        id="hokkaido_gen_001",
        name="苫東厚真発電所",
        capacity_mw=1650.0,
        fuel_type="coal",
        connected_bus_id="hokkaido_sub_001",
        region="hokkaido",
        latitude=42.6234,
        longitude=141.8567,
    )
    network.add_generator(gen)

    return network


@pytest.fixture
def two_region_networks(
    sample_grid_network: GridNetwork,
    sample_grid_network_50hz: GridNetwork,
) -> List[GridNetwork]:
    """Return a list of two regional networks for merge testing.

    Includes a Shikoku (60 Hz) and Hokkaido (50 Hz) network.
    """
    return [sample_grid_network, sample_grid_network_50hz]
