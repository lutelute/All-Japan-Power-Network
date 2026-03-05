"""Tests for interconnection data models and loader.

Tests cover:
- Interconnection dataclass validation (empty id, negative capacity, same from/to region)
- InterconnectionLoader loads all 9 records from YAML
- InterconnectionLoader raises FileNotFoundError for missing file
- InterconnectionFlow dataclass construction
- Interconnection and InterconnectionFlow exports from src.uc
"""

from pathlib import Path

import pytest

from src.uc.interconnection_loader import InterconnectionLoader
from src.uc.models import Interconnection, InterconnectionFlow
from tests.conftest import make_interconnection


# ======================================================================
# TestInterconnectionDataclass — validation
# ======================================================================


class TestInterconnectionDataclass:
    """Tests that the Interconnection dataclass validates inputs correctly."""

    def test_valid_interconnection(self) -> None:
        """A valid Interconnection instance is created without errors."""
        ic = make_interconnection()
        assert ic.id == "ic_test_001"
        assert ic.name_en == "Test Interconnection"
        assert ic.from_region == "tokyo"
        assert ic.to_region == "chubu"
        assert ic.capacity_mw == 1000.0
        assert ic.type == "AC"

    def test_empty_id_raises_value_error(self) -> None:
        """Empty id raises ValueError."""
        with pytest.raises(ValueError, match="id must not be empty"):
            make_interconnection(id="")

    def test_negative_capacity_raises_value_error(self) -> None:
        """Negative capacity_mw raises ValueError."""
        with pytest.raises(ValueError, match="capacity_mw must be positive"):
            make_interconnection(capacity_mw=-100.0)

    def test_zero_capacity_raises_value_error(self) -> None:
        """Zero capacity_mw raises ValueError."""
        with pytest.raises(ValueError, match="capacity_mw must be positive"):
            make_interconnection(capacity_mw=0.0)

    def test_same_from_to_region_raises_value_error(self) -> None:
        """Same from_region and to_region raises ValueError."""
        with pytest.raises(ValueError, match="from_region and to_region must differ"):
            make_interconnection(from_region="tokyo", to_region="tokyo")

    def test_custom_type_hvdc(self) -> None:
        """HVDC type is accepted."""
        ic = make_interconnection(type="HVDC")
        assert ic.type == "HVDC"

    def test_custom_type_fc(self) -> None:
        """FC (frequency converter) type is accepted."""
        ic = make_interconnection(type="FC")
        assert ic.type == "FC"

    def test_default_type_is_ac(self) -> None:
        """Default type is AC when not specified."""
        ic = Interconnection(
            id="ic_test",
            name_en="Test",
            from_region="hokkaido",
            to_region="tohoku",
            capacity_mw=500.0,
        )
        assert ic.type == "AC"


# ======================================================================
# TestInterconnectionFlow — dataclass construction
# ======================================================================


class TestInterconnectionFlow:
    """Tests for the InterconnectionFlow dataclass."""

    def test_default_construction(self) -> None:
        """InterconnectionFlow defaults to empty id and empty flow list."""
        flow = InterconnectionFlow()
        assert flow.interconnection_id == ""
        assert flow.flow_mw == []

    def test_construction_with_values(self) -> None:
        """InterconnectionFlow stores provided values correctly."""
        flow = InterconnectionFlow(
            interconnection_id="ic_001",
            flow_mw=[100.0, 200.0, -50.0],
        )
        assert flow.interconnection_id == "ic_001"
        assert flow.flow_mw == [100.0, 200.0, -50.0]
        assert len(flow.flow_mw) == 3


# ======================================================================
# TestInterconnectionLoader — YAML loading
# ======================================================================


class TestInterconnectionLoader:
    """Tests for the InterconnectionLoader class."""

    def test_load_all_9_interconnections(self, project_root: Path) -> None:
        """Loader reads all 9 interconnection records from YAML."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        assert len(interconnections) == 9

    def test_loaded_records_are_interconnection_instances(
        self, project_root: Path
    ) -> None:
        """All loaded records are Interconnection dataclass instances."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        for ic in interconnections:
            assert isinstance(ic, Interconnection)

    def test_first_record_is_hokkaido_honshu_hvdc(
        self, project_root: Path
    ) -> None:
        """First interconnection is the Hokkaido-Honshu HVDC link."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        ic = interconnections[0]
        assert ic.id == "ic_001"
        assert ic.name_en == "Hokkaido-Honshu HVDC Link"
        assert ic.from_region == "hokkaido"
        assert ic.to_region == "tohoku"
        assert ic.capacity_mw == 900.0
        assert ic.type == "HVDC"

    def test_tokyo_chubu_fc_interconnection(self, project_root: Path) -> None:
        """Tokyo-Chubu FC interconnection has correct capacity and type."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        # ic_003 is Tokyo-Chubu FC
        ic_003 = [ic for ic in interconnections if ic.id == "ic_003"][0]
        assert ic_003.from_region == "tokyo"
        assert ic_003.to_region == "chubu"
        assert ic_003.capacity_mw == 2100.0
        assert ic_003.type == "FC"

    def test_all_ids_unique(self, project_root: Path) -> None:
        """All interconnection IDs are unique."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        ids = [ic.id for ic in interconnections]
        assert len(ids) == len(set(ids))

    def test_all_capacities_positive(self, project_root: Path) -> None:
        """All interconnection capacities are positive."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        for ic in interconnections:
            assert ic.capacity_mw > 0, f"{ic.id} has non-positive capacity"

    def test_all_from_to_regions_differ(self, project_root: Path) -> None:
        """No interconnection has the same from_region and to_region."""
        yaml_path = str(project_root / "data" / "reference" / "interconnections.yaml")
        loader = InterconnectionLoader()
        interconnections = loader.load(yaml_path)

        for ic in interconnections:
            assert ic.from_region != ic.to_region, (
                f"{ic.id} has same from/to region: {ic.from_region}"
            )

    def test_missing_file_raises_file_not_found_error(self) -> None:
        """Loader raises FileNotFoundError for a non-existent YAML path."""
        loader = InterconnectionLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/interconnections.yaml")


# ======================================================================
# TestPackageExports — public API
# ======================================================================


class TestPackageExports:
    """Tests that Interconnection and InterconnectionFlow are exported from src.uc."""

    def test_interconnection_importable_from_uc(self) -> None:
        """Interconnection is importable from the src.uc package."""
        from src.uc import Interconnection as IC

        ic = IC(
            id="test",
            name_en="Test",
            from_region="a",
            to_region="b",
            capacity_mw=100.0,
        )
        assert ic.id == "test"

    def test_interconnection_flow_importable_from_uc(self) -> None:
        """InterconnectionFlow is importable from the src.uc package."""
        from src.uc import InterconnectionFlow as ICFlow

        flow = ICFlow(interconnection_id="test", flow_mw=[1.0, 2.0])
        assert flow.interconnection_id == "test"
