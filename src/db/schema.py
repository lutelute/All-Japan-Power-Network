"""SQLAlchemy 2.0+ ORM schema for grid attribute storage.

Defines four tables:

- **generator_attributes** — Mutable generator properties (fuel type,
  capacity, cost parameters, storage characteristics).
- **substation_attributes** — Mutable substation/bus properties
  (tap ratio, voltage setpoint, zone assignment).
- **load_attributes** — Mutable load properties (load model type,
  power factor, scaling factors).
- **schema_version** — Tracks the current database schema version
  for lightweight migration support.

All tables use SQLAlchemy 2.0 ``DeclarativeBase`` with ``Mapped``
type annotations and ``mapped_column()`` column definitions.

Usage::

    from sqlalchemy import create_engine
    from src.db.schema import Base, GeneratorAttributes

    engine = create_engine("sqlite:///data/grid_attributes.db")
    Base.metadata.create_all(engine)
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 declarative base for all grid attribute tables."""

    pass


class GeneratorAttributes(Base):
    """Mutable attributes for a generator (発電所).

    Stores cost parameters, operational limits, and storage
    characteristics that may be updated independently of the static
    network topology extracted from GIS sources.

    Mirrors the editable subset of fields from
    :class:`src.model.generator.Generator`, enabling database-backed
    attribute overrides without rebuilding the full network.

    Attributes:
        id: Generator identifier matching ``Generator.id``.
        fuel_type: Fuel type string (e.g. ``'coal'``, ``'lng'``).
        capacity_mw: Rated generation capacity in megawatts.
        p_min_mw: Minimum generation output in megawatts.
        vm_pu: Voltage magnitude setpoint in per-unit.
        status: Operational status (e.g. ``'active'``).
        startup_cost: Start-up cost (currency units).
        shutdown_cost: Shut-down cost (currency units).
        min_up_time_h: Minimum on-time once started (hours).
        min_down_time_h: Minimum off-time once shut down (hours).
        ramp_up_mw_per_h: Maximum ramp-up rate (MW/h).
        ramp_down_mw_per_h: Maximum ramp-down rate (MW/h).
        fuel_cost_per_mwh: Fuel cost per MWh (currency units).
        labor_cost_per_h: Labor cost per hour of operation.
        no_load_cost: Fixed on-state cost regardless of output.
        storage_capacity_mwh: Energy storage capacity (MWh).
        charge_rate_mw: Maximum charge rate (MW).
        discharge_rate_mw: Maximum discharge rate (MW).
        charge_efficiency: Round-trip charge efficiency (0–1].
        discharge_efficiency: Round-trip discharge efficiency (0–1].
        updated_at: Timestamp of last modification (UTC).
    """

    __tablename__ = "generator_attributes"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    fuel_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    capacity_mw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p_min_mw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vm_pu: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Unit commitment cost parameters
    startup_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    shutdown_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    min_up_time_h: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    min_down_time_h: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ramp_up_mw_per_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    ramp_down_mw_per_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    fuel_cost_per_mwh: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    labor_cost_per_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    no_load_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Storage parameters
    storage_capacity_mwh: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    charge_rate_mw: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    discharge_rate_mw: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    charge_efficiency: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    discharge_efficiency: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    # Metadata
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"GeneratorAttributes(id={self.id!r}, fuel_type={self.fuel_type!r}, "
            f"capacity_mw={self.capacity_mw})"
        )


class SubstationAttributes(Base):
    """Mutable attributes for a substation (変電所) / bus node.

    Stores voltage control parameters, tap ratios, and zone assignments
    that may be updated independently of the static GIS-sourced topology.

    Attributes:
        id: Substation identifier matching ``Substation.id``.
        voltage_setpoint_pu: Target voltage magnitude in per-unit.
        tap_ratio: Transformer tap ratio (1.0 = nominal).
        tap_min: Minimum tap ratio.
        tap_max: Maximum tap ratio.
        tap_step_percent: Tap step size as a percentage.
        zone: Zone assignment for multi-area studies.
        grid_class: Grid hierarchy classification
            (e.g. ``'backbone'``, ``'regional'``).
        status: Operational status (e.g. ``'active'``).
        updated_at: Timestamp of last modification (UTC).
    """

    __tablename__ = "substation_attributes"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    voltage_setpoint_pu: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    tap_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tap_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tap_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tap_step_percent: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    zone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    grid_class: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Metadata
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"SubstationAttributes(id={self.id!r}, "
            f"voltage_setpoint_pu={self.voltage_setpoint_pu}, "
            f"zone={self.zone!r})"
        )


class LoadAttributes(Base):
    """Mutable attributes for a load element.

    Stores load model classification, power factors, and scaling factors
    that may be updated independently of the static network topology.

    Attributes:
        id: Load identifier (e.g. ``'{region}_load_{bus_id}'``).
        bus_id: Associated bus/substation identifier.
        load_model: Load model type
            (e.g. ``'constant_power'``, ``'constant_impedance'``,
            ``'zip'``).
        p_mw: Active power demand in megawatts.
        q_mvar: Reactive power demand in megavar.
        power_factor: Power factor (cos phi) in (0, 1].
        scaling_factor: Multiplier applied to base demand (default 1.0).
        in_service: Whether this load is active.
        source: Data source identifier for traceability.
        updated_at: Timestamp of last modification (UTC).
    """

    __tablename__ = "load_attributes"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    bus_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    load_model: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    p_mw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    q_mvar: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    power_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scaling_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    in_service: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, default=1
    )
    source: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Metadata
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"LoadAttributes(id={self.id!r}, bus_id={self.bus_id!r}, "
            f"p_mw={self.p_mw}, load_model={self.load_model!r})"
        )


class SchemaVersion(Base):
    """Schema version tracking for lightweight migrations.

    Each row records a migration that has been applied to the database.
    The highest ``version`` number represents the current schema state.

    Attributes:
        version: Monotonically increasing schema version number.
        description: Human-readable description of the migration.
        applied_at: Timestamp when the migration was applied (UTC).
    """

    __tablename__ = "schema_version"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    applied_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"SchemaVersion(version={self.version}, "
            f"description={self.description!r})"
        )
