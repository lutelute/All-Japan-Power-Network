"""Data models for substations and shared enumerations.

Defines the core enumerations (VoltageClass, CapacityStatus, FuelType, BusType)
used across the pipeline, and the Substation dataclass representing a bus node
in the power grid network.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VoltageClass(Enum):
    """Transmission line voltage classification derived from KML line thickness.

    Maps KML visual style (line width) to standard Japanese voltage levels.
    REGIONAL (0) indicates voltage varies by region and must be resolved
    using the OCCTO voltage hierarchy in config/regions.yaml.
    """

    KV_500 = 500
    KV_275 = 275
    KV_220 = 220
    KV_187 = 187
    KV_154 = 154
    KV_132 = 132
    KV_110 = 110
    KV_77 = 77
    KV_66 = 66
    REGIONAL = 0  # Voltage varies by region — resolve via OCCTO hierarchy

    @classmethod
    def from_kv(cls, voltage_kv: float) -> "VoltageClass":
        """Look up a VoltageClass by its numeric kV value.

        Args:
            voltage_kv: Voltage in kilovolts.

        Returns:
            Matching VoltageClass member, or REGIONAL if no exact match.
        """
        rounded = int(round(voltage_kv))
        for member in cls:
            if member.value == rounded:
                return member
        return cls.REGIONAL


class CapacityStatus(Enum):
    """Transmission capacity status derived from KML line color.

    Maps KML visual style (line color) to grid capacity classifications
    per OCCTO open-capacity publication conventions.
    """

    ZERO_N1_INELIGIBLE = "zero_capacity_n1_ineligible"  # Red: no capacity, N-1 ineligible
    ZERO_N1_ELIGIBLE = "zero_capacity_n1_eligible"      # Orange: no capacity, N-1 eligible
    AVAILABLE = "available_capacity"                      # Blue: available capacity
    UNKNOWN = "unknown"                                   # Unmapped or missing style


class FuelType(Enum):
    """Power plant fuel type classification.

    Covers standard Japanese generation fuel categories as found in
    the 国土数値情報 P03 dataset.
    """

    COAL = "coal"
    LNG = "lng"
    OIL = "oil"
    NUCLEAR = "nuclear"
    HYDRO = "hydro"
    PUMPED_HYDRO = "pumped_hydro"
    GEOTHERMAL = "geothermal"
    WIND = "wind"
    SOLAR = "solar"
    BIOMASS = "biomass"
    MIXED = "mixed"
    UNKNOWN = "unknown"

    @classmethod
    def from_japanese(cls, name: str) -> "FuelType":
        """Resolve FuelType from a Japanese fuel name string.

        Args:
            name: Japanese fuel type name (e.g., '石炭', 'LNG', '原子力').

        Returns:
            Matching FuelType member, or UNKNOWN if not recognized.
        """
        mapping = {
            "石炭": cls.COAL,
            "coal": cls.COAL,
            "LNG": cls.LNG,
            "液化天然ガス": cls.LNG,
            "天然ガス": cls.LNG,
            "石油": cls.OIL,
            "重油": cls.OIL,
            "oil": cls.OIL,
            "原子力": cls.NUCLEAR,
            "nuclear": cls.NUCLEAR,
            "水力": cls.HYDRO,
            "hydro": cls.HYDRO,
            "揚水": cls.PUMPED_HYDRO,
            "pumped_hydro": cls.PUMPED_HYDRO,
            "地熱": cls.GEOTHERMAL,
            "geothermal": cls.GEOTHERMAL,
            "風力": cls.WIND,
            "wind": cls.WIND,
            "太陽光": cls.SOLAR,
            "solar": cls.SOLAR,
            "バイオマス": cls.BIOMASS,
            "biomass": cls.BIOMASS,
            "混合": cls.MIXED,
            "mixed": cls.MIXED,
        }
        return mapping.get(name.strip(), cls.UNKNOWN)


class BusType(Enum):
    """MATPOWER bus type classification.

    Uses MATPOWER convention:
        PQ (1)   — Load bus (default).
        PV (2)   — Generator bus with voltage setpoint.
        SLACK (3) — Reference/slack bus.

    In pandapower, bus types are determined implicitly by connected elements:
        PQ  → default bus (no generator or ext_grid)
        PV  → bus with generator (create_gen with vm_pu)
        SLACK → bus with ext_grid (create_ext_grid)
    The bus_type field in Substation is primarily for MATPOWER export.
    """

    PQ = 1      # Load bus
    PV = 2      # Generator bus
    SLACK = 3   # Reference bus


@dataclass
class Substation:
    """A substation (変電所) representing a bus node in the power grid.

    Extracted from KML Point features. Each substation maps to a single bus
    in the MATPOWER bus table and a single pandapower bus element.

    Attributes:
        id: Unique identifier, formatted as ``{region}_sub_{sequence}``.
        name: Substation name (Japanese, normalized via name_normalizer).
        region: Region identifier (e.g., 'hokkaido', 'tohoku').
        latitude: WGS-84 latitude in decimal degrees.
        longitude: WGS-84 longitude in decimal degrees.
        voltage_kv: Nominal voltage in kilovolts (e.g., 275.0, 500.0).
        bus_type: MATPOWER bus type — PQ (1), PV (2), or Slack (3).
        voltage_class: Classified voltage level from KML style.
        source_map: Source KML filename for traceability.
        grid_class: Grid hierarchy classification (e.g., 'backbone', 'regional').
        description: Optional description or notes from the KML source.
    """

    id: str
    name: str
    region: str
    latitude: float
    longitude: float
    voltage_kv: float
    bus_type: int = BusType.PQ.value
    voltage_class: Optional[VoltageClass] = None
    source_map: str = ""
    grid_class: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not self.id:
            raise ValueError("Substation id must not be empty")
        if not self.name:
            raise ValueError("Substation name must not be empty")
        if not self.region:
            raise ValueError("Substation region must not be empty")
        if self.voltage_kv < 0:
            raise ValueError(
                f"Substation voltage_kv must be non-negative, got {self.voltage_kv}"
            )

        # Auto-derive voltage_class from voltage_kv if not explicitly set
        if self.voltage_class is None and self.voltage_kv > 0:
            self.voltage_class = VoltageClass.from_kv(self.voltage_kv)

    @property
    def is_slack(self) -> bool:
        """Check if this substation is designated as a slack bus."""
        return self.bus_type == BusType.SLACK.value

    @property
    def is_generator_bus(self) -> bool:
        """Check if this substation is a PV (generator) bus."""
        return self.bus_type == BusType.PV.value

    @property
    def geodata(self) -> tuple:
        """Return pandapower-compatible geodata tuple (longitude, latitude).

        pandapower convention: geodata = (x, y) where x=longitude, y=latitude.
        """
        return (self.longitude, self.latitude)
