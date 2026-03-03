"""Data model for transmission lines (送電線).

Defines the TransmissionLine dataclass representing a branch (edge) in the
power grid network, connecting two substations with electrical parameters,
voltage class, and capacity status.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.model.substation import CapacityStatus, VoltageClass


@dataclass
class TransmissionLine:
    """A transmission line (送電線) representing a branch in the power grid.

    Extracted from KML LineString features. Each transmission line maps to a
    single branch in the MATPOWER branch table and a single pandapower line
    element created via ``create_line_from_parameters()``.

    Attributes:
        id: Unique identifier, formatted as ``{region}_line_{sequence}``.
        name: Line name (Japanese, normalized via name_normalizer).
        from_substation_id: ID of the originating substation (bus).
        to_substation_id: ID of the destination substation (bus).
        voltage_kv: Nominal voltage in kilovolts (e.g., 275.0, 500.0).
        length_km: Line length in kilometers, computed from coordinates
            using Haversine formula.
        region: Region identifier (e.g., 'hokkaido', 'tohoku').
        r_ohm_per_km: Resistance per kilometer (Ω/km).
        x_ohm_per_km: Reactance per kilometer (Ω/km).
        c_nf_per_km: Capacitance per kilometer (nF/km) for pandapower.
            Converted from susceptance B (S/km) via
            ``c_nf = B / (2 * π * f_hz) * 1e9``.
        max_i_ka: Maximum current rating in kiloamperes.
        capacity_status: Transmission capacity status from KML line color.
        voltage_class: Classified voltage level from KML line thickness.
        n1_eligible: Whether the line is eligible for N-1 contingency control.
        grid_class: Grid hierarchy classification (e.g., 'backbone', 'regional').
        coordinates: Ordered list of (latitude, longitude) waypoints from KML.
        source_map: Source KML filename for traceability.
        description: Optional description or notes from the KML source.
    """

    # Required fields
    id: str
    name: str
    from_substation_id: str
    to_substation_id: str
    voltage_kv: float
    length_km: float

    # Optional fields with defaults
    region: str = ""
    r_ohm_per_km: float = 0.0
    x_ohm_per_km: float = 0.0
    c_nf_per_km: float = 0.0
    max_i_ka: float = 0.0
    capacity_status: CapacityStatus = CapacityStatus.UNKNOWN
    voltage_class: Optional[VoltageClass] = None
    n1_eligible: bool = False
    grid_class: str = ""
    coordinates: List[Tuple[float, float]] = field(default_factory=list)
    source_map: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not self.id:
            raise ValueError("TransmissionLine id must not be empty")
        if not self.name:
            raise ValueError("TransmissionLine name must not be empty")
        if not self.from_substation_id:
            raise ValueError("TransmissionLine from_substation_id must not be empty")
        if not self.to_substation_id:
            raise ValueError("TransmissionLine to_substation_id must not be empty")
        if self.voltage_kv < 0:
            raise ValueError(
                f"TransmissionLine voltage_kv must be non-negative, got {self.voltage_kv}"
            )
        if self.length_km < 0:
            raise ValueError(
                f"TransmissionLine length_km must be non-negative, got {self.length_km}"
            )

        # Auto-derive voltage_class from voltage_kv if not explicitly set
        if self.voltage_class is None and self.voltage_kv > 0:
            self.voltage_class = VoltageClass.from_kv(self.voltage_kv)

        # Derive n1_eligible from capacity_status if applicable
        if self.capacity_status == CapacityStatus.ZERO_N1_ELIGIBLE:
            self.n1_eligible = True

    @property
    def has_electrical_parameters(self) -> bool:
        """Check if electrical parameters have been populated."""
        return self.r_ohm_per_km > 0 or self.x_ohm_per_km > 0

    @property
    def is_backbone(self) -> bool:
        """Check if this line belongs to the backbone grid (500kV or 275kV)."""
        return self.voltage_kv >= 275.0

    @property
    def endpoint_ids(self) -> Tuple[str, str]:
        """Return the from/to substation IDs as a tuple."""
        return (self.from_substation_id, self.to_substation_id)
