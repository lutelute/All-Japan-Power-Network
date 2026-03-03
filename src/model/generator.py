"""Data model for generators (発電所).

Defines the Generator dataclass representing a power generation facility
in the grid network, with capacity, fuel type, and location data sourced
primarily from the 国土数値情報 P03 dataset.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.model.substation import FuelType


@dataclass
class Generator:
    """A generator (発電所) representing a power plant in the grid.

    Extracted from 国土数値情報 P03 dataset and matched to the nearest
    substation bus via geographic proximity. Each generator maps to a
    single entry in the MATPOWER gen table and a pandapower gen element
    created via ``create_gen()``.

    Attributes:
        id: Unique identifier, formatted as ``{region}_gen_{sequence}``.
        name: Generator/power plant name (Japanese).
        capacity_mw: Rated generation capacity in megawatts.
        fuel_type: Fuel type classification (coal, lng, nuclear, etc.).
            Accepts either a FuelType enum or a string value.
        connected_bus_id: ID of the substation bus this generator is
            connected to. Determined by nearest-substation matching.
        region: Region identifier (e.g., 'hokkaido', 'tohoku').
        latitude: WGS-84 latitude of the power plant in decimal degrees.
        longitude: WGS-84 longitude of the power plant in decimal degrees.
        operator: Operating company name (事業者名).
        status: Operational status (e.g., 'active', 'decommissioned', 'planned').
        vm_pu: Voltage magnitude setpoint in per-unit for PV bus control.
        p_min_mw: Minimum generation output in megawatts.
        source: Data source identifier for traceability.
        description: Optional description or notes.
        startup_cost: Cost incurred when starting up the generator (currency units).
        shutdown_cost: Cost incurred when shutting down the generator (currency units).
        min_up_time_h: Minimum number of hours the generator must remain on
            once started.
        min_down_time_h: Minimum number of hours the generator must remain off
            once shut down.
        ramp_up_mw_per_h: Maximum ramp-up rate in MW per hour. None means
            unlimited (instant ramp-up).
        ramp_down_mw_per_h: Maximum ramp-down rate in MW per hour. None means
            unlimited (instant ramp-down).
        fuel_cost_per_mwh: Fuel cost per MWh of generation (currency units).
        labor_cost_per_h: Labor cost per hour of operation (currency units).
        no_load_cost: Fixed cost incurred while the generator is on, regardless
            of output level (currency units).
        maintenance_windows: List of (start_hour, end_hour) tuples indicating
            scheduled maintenance periods when the generator is unavailable.
        construction_date: Date the generator was constructed (ISO 8601 string).
        rebuild_planned_date: Planned date for rebuild or major overhaul
            (ISO 8601 string).
        disaster_risk_score: Risk score for natural disaster vulnerability
            (0.0 = no risk, higher = more risk).
        storage_capacity_mwh: Energy storage capacity in megawatt-hours.
            A value > 0 indicates this generator has storage capability.
        charge_rate_mw: Maximum charging rate in megawatts. None means
            the rate equals capacity_mw.
        discharge_rate_mw: Maximum discharging rate in megawatts. None means
            the rate equals capacity_mw.
        charge_efficiency: Round-trip charging efficiency as a fraction
            in (0, 1]. Default is 0.90.
        discharge_efficiency: Round-trip discharging efficiency as a fraction
            in (0, 1]. Default is 0.90.
        initial_soc_fraction: Initial state-of-charge as a fraction of
            storage_capacity_mwh in [0, 1]. Default is 0.5.
        min_terminal_soc_fraction: Minimum terminal state-of-charge as a
            fraction of storage_capacity_mwh in [0, 1]. Default is 0.5.
    """

    # Required fields
    id: str
    name: str
    capacity_mw: float
    fuel_type: str  # Accepts string; validated in __post_init__

    # Optional fields with defaults
    connected_bus_id: str = ""
    region: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    operator: str = ""
    status: str = "active"
    vm_pu: float = 1.0
    p_min_mw: float = 0.0
    source: str = ""
    description: str = ""

    # Unit commitment (UC) fields
    startup_cost: float = 0.0
    shutdown_cost: float = 0.0
    min_up_time_h: int = 1
    min_down_time_h: int = 1
    ramp_up_mw_per_h: Optional[float] = None  # None = unlimited
    ramp_down_mw_per_h: Optional[float] = None  # None = unlimited
    fuel_cost_per_mwh: float = 0.0
    labor_cost_per_h: float = 0.0
    no_load_cost: float = 0.0
    maintenance_windows: List[Tuple[int, int]] = field(default_factory=list)
    construction_date: Optional[str] = None
    rebuild_planned_date: Optional[str] = None
    disaster_risk_score: float = 0.0

    # Storage fields
    storage_capacity_mwh: float = 0.0
    charge_rate_mw: Optional[float] = None
    discharge_rate_mw: Optional[float] = None
    charge_efficiency: float = 0.90
    discharge_efficiency: float = 0.90
    initial_soc_fraction: float = 0.5
    min_terminal_soc_fraction: float = 0.5

    # Resolved enum (set in __post_init__)
    _fuel_type_enum: Optional[FuelType] = None

    def __post_init__(self) -> None:
        """Validate fields and resolve fuel_type to enum."""
        if not self.id:
            raise ValueError("Generator id must not be empty")
        if not self.name:
            raise ValueError("Generator name must not be empty")
        if self.capacity_mw < 0:
            raise ValueError(
                f"Generator capacity_mw must be non-negative, got {self.capacity_mw}"
            )
        if self.vm_pu <= 0:
            raise ValueError(
                f"Generator vm_pu must be positive, got {self.vm_pu}"
            )

        # Validate UC-specific fields
        if self.startup_cost < 0:
            raise ValueError(
                f"Generator startup_cost must be non-negative, got {self.startup_cost}"
            )
        if self.shutdown_cost < 0:
            raise ValueError(
                f"Generator shutdown_cost must be non-negative, got {self.shutdown_cost}"
            )
        if self.min_up_time_h < 1:
            raise ValueError(
                f"Generator min_up_time_h must be >= 1, got {self.min_up_time_h}"
            )
        if self.min_down_time_h < 1:
            raise ValueError(
                f"Generator min_down_time_h must be >= 1, got {self.min_down_time_h}"
            )
        if self.ramp_up_mw_per_h is not None and self.ramp_up_mw_per_h < 0:
            raise ValueError(
                f"Generator ramp_up_mw_per_h must be non-negative, got {self.ramp_up_mw_per_h}"
            )
        if self.ramp_down_mw_per_h is not None and self.ramp_down_mw_per_h < 0:
            raise ValueError(
                f"Generator ramp_down_mw_per_h must be non-negative, got {self.ramp_down_mw_per_h}"
            )
        if self.fuel_cost_per_mwh < 0:
            raise ValueError(
                f"Generator fuel_cost_per_mwh must be non-negative, got {self.fuel_cost_per_mwh}"
            )
        if self.labor_cost_per_h < 0:
            raise ValueError(
                f"Generator labor_cost_per_h must be non-negative, got {self.labor_cost_per_h}"
            )
        if self.no_load_cost < 0:
            raise ValueError(
                f"Generator no_load_cost must be non-negative, got {self.no_load_cost}"
            )
        if self.disaster_risk_score < 0:
            raise ValueError(
                f"Generator disaster_risk_score must be non-negative, got {self.disaster_risk_score}"
            )
        for window in self.maintenance_windows:
            if not isinstance(window, tuple) or len(window) != 2:
                raise ValueError(
                    f"Each maintenance window must be a (start, end) tuple, got {window}"
                )
            if window[0] >= window[1]:
                raise ValueError(
                    f"Maintenance window start must be < end, got {window}"
                )

        # Validate storage fields
        if self.storage_capacity_mwh < 0:
            raise ValueError(
                f"Generator storage_capacity_mwh must be non-negative, got {self.storage_capacity_mwh}"
            )
        if self.charge_rate_mw is not None and self.charge_rate_mw < 0:
            raise ValueError(
                f"Generator charge_rate_mw must be non-negative, got {self.charge_rate_mw}"
            )
        if self.discharge_rate_mw is not None and self.discharge_rate_mw < 0:
            raise ValueError(
                f"Generator discharge_rate_mw must be non-negative, got {self.discharge_rate_mw}"
            )
        if not (0 < self.charge_efficiency <= 1):
            raise ValueError(
                f"Generator charge_efficiency must be in (0, 1], got {self.charge_efficiency}"
            )
        if not (0 < self.discharge_efficiency <= 1):
            raise ValueError(
                f"Generator discharge_efficiency must be in (0, 1], got {self.discharge_efficiency}"
            )
        if not (0 <= self.initial_soc_fraction <= 1):
            raise ValueError(
                f"Generator initial_soc_fraction must be in [0, 1], got {self.initial_soc_fraction}"
            )
        if not (0 <= self.min_terminal_soc_fraction <= 1):
            raise ValueError(
                f"Generator min_terminal_soc_fraction must be in [0, 1], got {self.min_terminal_soc_fraction}"
            )

        # Resolve fuel_type string to FuelType enum
        self._fuel_type_enum = self._resolve_fuel_type(self.fuel_type)

    @staticmethod
    def _resolve_fuel_type(fuel_type_str: str) -> FuelType:
        """Resolve a fuel type string to a FuelType enum member.

        Tries direct enum value match first, then falls back to
        FuelType.from_japanese() for Japanese name resolution.

        Args:
            fuel_type_str: Fuel type as string (English or Japanese).

        Returns:
            Resolved FuelType enum member.
        """
        # Try direct enum value match (e.g., 'coal', 'lng')
        for member in FuelType:
            if member.value == fuel_type_str.strip().lower():
                return member
        # Fall back to Japanese name resolution
        return FuelType.from_japanese(fuel_type_str)

    @property
    def fuel_type_enum(self) -> FuelType:
        """Return the resolved FuelType enum member."""
        if self._fuel_type_enum is None:
            self._fuel_type_enum = self._resolve_fuel_type(self.fuel_type)
        return self._fuel_type_enum

    @property
    def has_location(self) -> bool:
        """Check if geographic coordinates are available."""
        return self.latitude != 0.0 or self.longitude != 0.0

    @property
    def is_connected(self) -> bool:
        """Check if this generator is connected to a substation bus."""
        return bool(self.connected_bus_id)

    @property
    def is_storage(self) -> bool:
        """Check if this generator has energy storage capability."""
        return self.storage_capacity_mwh > 0

    @property
    def geodata(self) -> tuple:
        """Return pandapower-compatible geodata tuple (longitude, latitude).

        pandapower convention: geodata = (x, y) where x=longitude, y=latitude.
        """
        return (self.longitude, self.latitude)

    @property
    def is_renewable(self) -> bool:
        """Check if this generator uses a renewable fuel source."""
        renewable_types = {
            FuelType.HYDRO,
            FuelType.WIND,
            FuelType.SOLAR,
            FuelType.GEOTHERMAL,
            FuelType.BIOMASS,
        }
        return self.fuel_type_enum in renewable_types
