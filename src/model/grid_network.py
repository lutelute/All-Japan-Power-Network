"""Container model for a regional or national power grid network.

Defines the GridNetwork dataclass that aggregates substations, transmission
lines, and generators for a single region or a merged national model.
Supports adding elements, looking up by ID, merging multiple regional
networks, and computing summary statistics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.model.generator import Generator
from src.model.substation import Substation
from src.model.transmission_line import TransmissionLine


@dataclass
class GridNetwork:
    """A container aggregating all power grid elements for a region or nation.

    Each regional pipeline stage produces a GridNetwork for its region.
    Regional networks can be merged into a unified national GridNetwork
    via :meth:`merge` or the class method :meth:`merge_regions`.

    Attributes:
        region: Region identifier (e.g., 'hokkaido', 'tohoku') or
            'national' for the merged all-Japan model.
        frequency_hz: System frequency in hertz (50 or 60). Set to 0
            for the national model which spans both frequency zones.
        substations: List of Substation instances (bus nodes).
        transmission_lines: List of TransmissionLine instances (branches).
        generators: List of Generator instances (power plants).
        source_regions: List of region identifiers that contributed to
            this network (populated during merging).
        metadata: Free-form metadata dict for traceability and notes.
    """

    # Required fields
    region: str
    frequency_hz: int

    # Element collections
    substations: List[Substation] = field(default_factory=list)
    transmission_lines: List[TransmissionLine] = field(default_factory=list)
    generators: List[Generator] = field(default_factory=list)

    # Merge tracking
    source_regions: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, str] = field(default_factory=dict)

    # Internal lookup caches (rebuilt on demand)
    _substation_index: Dict[str, Substation] = field(
        default_factory=dict, repr=False
    )
    _generator_index: Dict[str, Generator] = field(
        default_factory=dict, repr=False
    )
    _line_index: Dict[str, TransmissionLine] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not self.region:
            raise ValueError("GridNetwork region must not be empty")
        if self.frequency_hz not in (0, 50, 60):
            raise ValueError(
                f"GridNetwork frequency_hz must be 0, 50, or 60, "
                f"got {self.frequency_hz}"
            )

        # Build initial lookup indices from any pre-populated lists
        self._rebuild_indices()

    # ------------------------------------------------------------------
    # Element addition
    # ------------------------------------------------------------------

    def add_substation(self, substation: Substation) -> None:
        """Add a substation to the network.

        Args:
            substation: Substation instance to add.

        Raises:
            ValueError: If a substation with the same ID already exists.
        """
        if substation.id in self._substation_index:
            raise ValueError(
                f"Duplicate substation id '{substation.id}' in "
                f"region '{self.region}'"
            )
        self.substations.append(substation)
        self._substation_index[substation.id] = substation

    def add_transmission_line(self, line: TransmissionLine) -> None:
        """Add a transmission line to the network.

        Args:
            line: TransmissionLine instance to add.

        Raises:
            ValueError: If a line with the same ID already exists.
        """
        if line.id in self._line_index:
            raise ValueError(
                f"Duplicate transmission line id '{line.id}' in "
                f"region '{self.region}'"
            )
        self.transmission_lines.append(line)
        self._line_index[line.id] = line

    def add_generator(self, generator: Generator) -> None:
        """Add a generator to the network.

        Args:
            generator: Generator instance to add.

        Raises:
            ValueError: If a generator with the same ID already exists.
        """
        if generator.id in self._generator_index:
            raise ValueError(
                f"Duplicate generator id '{generator.id}' in "
                f"region '{self.region}'"
            )
        self.generators.append(generator)
        self._generator_index[generator.id] = generator

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_substation(self, substation_id: str) -> Optional[Substation]:
        """Look up a substation by its unique ID.

        Args:
            substation_id: The substation identifier.

        Returns:
            The matching Substation, or None if not found.
        """
        return self._substation_index.get(substation_id)

    def get_transmission_line(self, line_id: str) -> Optional[TransmissionLine]:
        """Look up a transmission line by its unique ID.

        Args:
            line_id: The transmission line identifier.

        Returns:
            The matching TransmissionLine, or None if not found.
        """
        return self._line_index.get(line_id)

    def get_generator(self, generator_id: str) -> Optional[Generator]:
        """Look up a generator by its unique ID.

        Args:
            generator_id: The generator identifier.

        Returns:
            The matching Generator, or None if not found.
        """
        return self._generator_index.get(generator_id)

    # ------------------------------------------------------------------
    # Convenience aliases (used by pandapower_builder and xml_exporter)
    # ------------------------------------------------------------------

    @property
    def lines(self) -> List[TransmissionLine]:
        """Alias for transmission_lines (used by downstream modules)."""
        return self.transmission_lines

    # ------------------------------------------------------------------
    # Summary / statistics
    # ------------------------------------------------------------------

    @property
    def substation_count(self) -> int:
        """Return the number of substations in the network."""
        return len(self.substations)

    @property
    def line_count(self) -> int:
        """Return the number of transmission lines in the network."""
        return len(self.transmission_lines)

    @property
    def generator_count(self) -> int:
        """Return the number of generators in the network."""
        return len(self.generators)

    @property
    def total_generation_mw(self) -> float:
        """Return total rated generation capacity in megawatts."""
        return sum(gen.capacity_mw for gen in self.generators)

    @property
    def is_national(self) -> bool:
        """Check if this is a merged national (multi-region) model."""
        return self.region == "national" or len(self.source_regions) > 1

    @property
    def has_elements(self) -> bool:
        """Check if the network contains any grid elements."""
        return bool(self.substations or self.transmission_lines or self.generators)

    def summary(self) -> Dict[str, object]:
        """Return a summary dict of the network for logging/reporting.

        Returns:
            Dict with region, frequency, element counts, and capacity.
        """
        return {
            "region": self.region,
            "frequency_hz": self.frequency_hz,
            "substations": self.substation_count,
            "transmission_lines": self.line_count,
            "generators": self.generator_count,
            "total_generation_mw": self.total_generation_mw,
            "source_regions": list(self.source_regions),
        }

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge(self, other: "GridNetwork") -> None:
        """Merge another GridNetwork's elements into this network.

        All substations, transmission lines, and generators from *other*
        are appended to this network.  Duplicate IDs raise ValueError.
        After merging, ``source_regions`` is updated and frequency is set
        to 0 (mixed) if the two networks differ in frequency.

        Args:
            other: The GridNetwork to merge into this one.

        Raises:
            ValueError: If any element ID in *other* already exists in
                this network.
        """
        # Merge substations
        for sub in other.substations:
            self.add_substation(sub)

        # Merge transmission lines
        for line in other.transmission_lines:
            self.add_transmission_line(line)

        # Merge generators
        for gen in other.generators:
            self.add_generator(gen)

        # Track source regions
        if other.region not in self.source_regions:
            self.source_regions.append(other.region)
        if self.region not in self.source_regions and self.region != "national":
            self.source_regions.insert(0, self.region)

        # Mixed frequency → set to 0
        if self.frequency_hz != other.frequency_hz and self.frequency_hz != 0:
            self.frequency_hz = 0

    @classmethod
    def merge_regions(cls, networks: List["GridNetwork"]) -> "GridNetwork":
        """Create a national GridNetwork by merging multiple regional networks.

        Args:
            networks: List of regional GridNetwork instances to merge.

        Returns:
            A new GridNetwork with region='national' containing all
            elements from all input networks.

        Raises:
            ValueError: If *networks* is empty or contains duplicate
                element IDs across regions.
        """
        if not networks:
            raise ValueError("Cannot merge an empty list of networks")

        national = cls(region="national", frequency_hz=0)

        for network in networks:
            national.merge(network)

        return national

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_references(self) -> List[str]:
        """Check that all transmission line endpoints reference existing substations.

        Returns:
            List of warning messages for unresolved references.
            An empty list indicates all references are valid.
        """
        warnings: List[str] = []
        for line in self.transmission_lines:
            if line.from_substation_id not in self._substation_index:
                warnings.append(
                    f"Line '{line.id}' references unknown from_substation "
                    f"'{line.from_substation_id}'"
                )
            if line.to_substation_id not in self._substation_index:
                warnings.append(
                    f"Line '{line.id}' references unknown to_substation "
                    f"'{line.to_substation_id}'"
                )
        return warnings

    def get_isolated_substations(self) -> List[Substation]:
        """Find substations not connected to any transmission line.

        Returns:
            List of Substation instances with no connecting lines.
        """
        connected_ids: set = set()
        for line in self.transmission_lines:
            connected_ids.add(line.from_substation_id)
            connected_ids.add(line.to_substation_id)

        return [
            sub for sub in self.substations
            if sub.id not in connected_ids
        ]

    def get_isolated_generators(self) -> List[Generator]:
        """Find generators whose connected_bus_id does not match any substation.

        A generator is considered isolated if its ``connected_bus_id`` is
        empty or does not correspond to any substation ID in the network.

        Returns:
            List of Generator instances with unresolved bus connections.
        """
        return [
            gen for gen in self.generators
            if not gen.connected_bus_id
            or gen.connected_bus_id not in self._substation_index
        ]

    def get_orphaned_lines(self) -> List[TransmissionLine]:
        """Find transmission lines with at least one unresolved endpoint.

        A line is considered orphaned if its ``from_substation_id`` or
        ``to_substation_id`` does not correspond to any substation ID
        in the network.

        Returns:
            List of TransmissionLine instances with unresolved endpoints.
        """
        return [
            line for line in self.transmission_lines
            if line.from_substation_id not in self._substation_index
            or line.to_substation_id not in self._substation_index
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_indices(self) -> None:
        """Rebuild internal lookup dicts from current element lists."""
        self._substation_index = {sub.id: sub for sub in self.substations}
        self._generator_index = {gen.id: gen for gen in self.generators}
        self._line_index = {line.id: line for line in self.transmission_lines}
