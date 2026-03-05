"""Unit tests for the database layer (CRUD, schema migrations, versioning).

Tests GridDatabase CRUD operations for generator, substation, and load
attributes; partial upsert behaviour (update without overwriting existing
fields); deletion; listing; schema version tracking; migration manager
application; and edge cases such as nonexistent record retrieval, invalid
attribute names, and duplicate upserts.
"""

import pytest
from sqlalchemy import create_engine, inspect, text

from src.db.grid_db import GridDatabase
from src.db.migrations import Migration, MigrationManager, MIGRATIONS
from src.db.schema import (
    Base,
    GeneratorAttributes,
    LoadAttributes,
    SchemaVersion,
    SubstationAttributes,
)


# ======================================================================
# Schema definition
# ======================================================================


class TestSchemaDefinition:
    """Tests for SQLAlchemy ORM schema table definitions."""

    def test_all_tables_defined(self) -> None:
        """Base metadata contains all four expected tables."""
        table_names = set(Base.metadata.tables.keys())
        assert "generator_attributes" in table_names
        assert "substation_attributes" in table_names
        assert "load_attributes" in table_names
        assert "schema_version" in table_names

    def test_generator_attributes_columns(self) -> None:
        """GeneratorAttributes table has key columns."""
        col_names = [c.name for c in GeneratorAttributes.__table__.columns]
        assert "id" in col_names
        assert "fuel_type" in col_names
        assert "capacity_mw" in col_names
        assert "fuel_cost_per_mwh" in col_names
        assert "startup_cost" in col_names
        assert "storage_capacity_mwh" in col_names
        assert "updated_at" in col_names

    def test_substation_attributes_columns(self) -> None:
        """SubstationAttributes table has key columns."""
        col_names = [c.name for c in SubstationAttributes.__table__.columns]
        assert "id" in col_names
        assert "voltage_setpoint_pu" in col_names
        assert "tap_ratio" in col_names
        assert "zone" in col_names
        assert "updated_at" in col_names

    def test_load_attributes_columns(self) -> None:
        """LoadAttributes table has key columns."""
        col_names = [c.name for c in LoadAttributes.__table__.columns]
        assert "id" in col_names
        assert "bus_id" in col_names
        assert "load_model" in col_names
        assert "p_mw" in col_names
        assert "power_factor" in col_names
        assert "updated_at" in col_names

    def test_schema_version_columns(self) -> None:
        """SchemaVersion table has key columns."""
        col_names = [c.name for c in SchemaVersion.__table__.columns]
        assert "version" in col_names
        assert "description" in col_names
        assert "applied_at" in col_names

    def test_generator_repr(self) -> None:
        """GeneratorAttributes repr includes id, fuel_type, capacity_mw."""
        gen = GeneratorAttributes(id="g1", fuel_type="coal", capacity_mw=500.0)
        r = repr(gen)
        assert "g1" in r
        assert "coal" in r
        assert "500.0" in r

    def test_substation_repr(self) -> None:
        """SubstationAttributes repr includes id."""
        sub = SubstationAttributes(id="s1", zone="shikoku")
        r = repr(sub)
        assert "s1" in r
        assert "shikoku" in r

    def test_load_repr(self) -> None:
        """LoadAttributes repr includes id and bus_id."""
        load = LoadAttributes(id="l1", bus_id="bus_a")
        r = repr(load)
        assert "l1" in r
        assert "bus_a" in r

    def test_schema_version_repr(self) -> None:
        """SchemaVersion repr includes version number."""
        sv = SchemaVersion(version=1, description="Initial")
        r = repr(sv)
        assert "1" in r
        assert "Initial" in r


# ======================================================================
# GridDatabase: initialisation
# ======================================================================


class TestGridDatabaseInit:
    """Tests for GridDatabase construction and initialisation."""

    def test_in_memory_creation(self) -> None:
        """In-memory database is created successfully."""
        db = GridDatabase(":memory:")
        assert db is not None

    def test_schema_version_after_init(self) -> None:
        """Schema version is >= 1 after initialisation."""
        db = GridDatabase(":memory:")
        version = db.get_schema_version()
        assert version >= 1

    def test_file_based_creation(self, tmp_path) -> None:
        """File-based SQLite database is created."""
        db_path = str(tmp_path / "test_grid.db")
        db = GridDatabase(db_path)
        assert db.get_schema_version() >= 1

    def test_reopen_preserves_data(self, tmp_path) -> None:
        """Reopening a file-based DB preserves previously stored data."""
        db_path = str(tmp_path / "persist_test.db")
        db1 = GridDatabase(db_path)
        db1.upsert_generator_attributes("gen_persist", fuel_type="lng")

        db2 = GridDatabase(db_path)
        attrs = db2.get_generator_attributes("gen_persist")
        assert attrs is not None
        assert attrs.fuel_type == "lng"


# ======================================================================
# GridDatabase: Generator attributes CRUD
# ======================================================================


class TestGeneratorCRUD:
    """Tests for generator attribute CRUD operations."""

    def test_create_and_read(self, empty_grid_db: GridDatabase) -> None:
        """Can store and retrieve generator attributes."""
        empty_grid_db.upsert_generator_attributes(
            "gen_1", fuel_type="coal", capacity_mw=500.0
        )
        attrs = empty_grid_db.get_generator_attributes("gen_1")
        assert attrs is not None
        assert attrs.id == "gen_1"
        assert attrs.fuel_type == "coal"
        assert attrs.capacity_mw == 500.0

    def test_read_nonexistent_returns_none(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """Reading a nonexistent record returns None."""
        attrs = empty_grid_db.get_generator_attributes("no_such_gen")
        assert attrs is None

    def test_partial_update_preserves_existing(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """Partial upsert updates only provided fields."""
        empty_grid_db.upsert_generator_attributes(
            "gen_1", fuel_type="coal", capacity_mw=500.0
        )
        # Update only fuel_cost — fuel_type and capacity_mw must persist
        empty_grid_db.upsert_generator_attributes(
            "gen_1", fuel_cost_per_mwh=25.0
        )
        attrs = empty_grid_db.get_generator_attributes("gen_1")
        assert attrs.fuel_type == "coal"
        assert attrs.capacity_mw == 500.0
        assert attrs.fuel_cost_per_mwh == 25.0

    def test_overwrite_existing_field(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """Upsert with same field name overwrites the old value."""
        empty_grid_db.upsert_generator_attributes(
            "gen_1", fuel_type="coal"
        )
        empty_grid_db.upsert_generator_attributes(
            "gen_1", fuel_type="lng"
        )
        attrs = empty_grid_db.get_generator_attributes("gen_1")
        assert attrs.fuel_type == "lng"

    def test_updated_at_set(self, empty_grid_db: GridDatabase) -> None:
        """updated_at timestamp is set on creation."""
        empty_grid_db.upsert_generator_attributes("gen_1", fuel_type="coal")
        attrs = empty_grid_db.get_generator_attributes("gen_1")
        assert attrs.updated_at is not None

    def test_updated_at_changes_on_update(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """updated_at timestamp changes on subsequent update."""
        empty_grid_db.upsert_generator_attributes("gen_1", fuel_type="coal")
        attrs1 = empty_grid_db.get_generator_attributes("gen_1")
        t1 = attrs1.updated_at

        empty_grid_db.upsert_generator_attributes("gen_1", capacity_mw=999.0)
        attrs2 = empty_grid_db.get_generator_attributes("gen_1")
        t2 = attrs2.updated_at
        assert t2 >= t1

    def test_delete_existing(self, empty_grid_db: GridDatabase) -> None:
        """Deleting an existing record returns True."""
        empty_grid_db.upsert_generator_attributes("gen_1", fuel_type="coal")
        result = empty_grid_db.delete_generator_attributes("gen_1")
        assert result is True
        assert empty_grid_db.get_generator_attributes("gen_1") is None

    def test_delete_nonexistent(self, empty_grid_db: GridDatabase) -> None:
        """Deleting a nonexistent record returns False."""
        result = empty_grid_db.delete_generator_attributes("no_such_gen")
        assert result is False

    def test_list_generators(self, empty_grid_db: GridDatabase) -> None:
        """list_generator_attributes returns all stored records."""
        empty_grid_db.upsert_generator_attributes("gen_1", fuel_type="coal")
        empty_grid_db.upsert_generator_attributes("gen_2", fuel_type="lng")
        records = empty_grid_db.list_generator_attributes()
        assert len(records) == 2
        ids = {r.id for r in records}
        assert ids == {"gen_1", "gen_2"}

    def test_list_generators_empty(self, empty_grid_db: GridDatabase) -> None:
        """list_generator_attributes returns empty list for empty DB."""
        records = empty_grid_db.list_generator_attributes()
        assert records == []

    def test_invalid_attribute_raises(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """Upsert with invalid attribute name raises AttributeError."""
        empty_grid_db.upsert_generator_attributes("gen_1", fuel_type="coal")
        with pytest.raises(AttributeError, match="no attribute"):
            empty_grid_db.upsert_generator_attributes(
                "gen_1", nonexistent_field=123
            )

    def test_storage_fields(self, empty_grid_db: GridDatabase) -> None:
        """Storage-related generator fields are stored correctly."""
        empty_grid_db.upsert_generator_attributes(
            "gen_storage",
            fuel_type="hydro",
            storage_capacity_mwh=400.0,
            charge_rate_mw=100.0,
            discharge_rate_mw=100.0,
            charge_efficiency=0.85,
            discharge_efficiency=0.90,
        )
        attrs = empty_grid_db.get_generator_attributes("gen_storage")
        assert attrs.storage_capacity_mwh == 400.0
        assert attrs.charge_rate_mw == 100.0
        assert attrs.discharge_rate_mw == 100.0
        assert attrs.charge_efficiency == 0.85
        assert attrs.discharge_efficiency == 0.90

    def test_uc_cost_fields(self, empty_grid_db: GridDatabase) -> None:
        """Unit commitment cost fields are stored correctly."""
        empty_grid_db.upsert_generator_attributes(
            "gen_uc",
            startup_cost=50000.0,
            shutdown_cost=10000.0,
            min_up_time_h=8,
            min_down_time_h=6,
            ramp_up_mw_per_h=50.0,
            ramp_down_mw_per_h=40.0,
            fuel_cost_per_mwh=25.0,
            labor_cost_per_h=100.0,
            no_load_cost=5000.0,
        )
        attrs = empty_grid_db.get_generator_attributes("gen_uc")
        assert attrs.startup_cost == 50000.0
        assert attrs.shutdown_cost == 10000.0
        assert attrs.min_up_time_h == 8
        assert attrs.min_down_time_h == 6
        assert attrs.ramp_up_mw_per_h == 50.0
        assert attrs.ramp_down_mw_per_h == 40.0
        assert attrs.fuel_cost_per_mwh == 25.0
        assert attrs.labor_cost_per_h == 100.0
        assert attrs.no_load_cost == 5000.0


# ======================================================================
# GridDatabase: Substation attributes CRUD
# ======================================================================


class TestSubstationCRUD:
    """Tests for substation attribute CRUD operations."""

    def test_create_and_read(self, empty_grid_db: GridDatabase) -> None:
        """Can store and retrieve substation attributes."""
        empty_grid_db.upsert_substation_attributes(
            "sub_1",
            voltage_setpoint_pu=1.02,
            tap_ratio=1.0,
            zone="shikoku",
        )
        attrs = empty_grid_db.get_substation_attributes("sub_1")
        assert attrs is not None
        assert attrs.id == "sub_1"
        assert attrs.voltage_setpoint_pu == 1.02
        assert attrs.tap_ratio == 1.0
        assert attrs.zone == "shikoku"

    def test_read_nonexistent_returns_none(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """Reading a nonexistent substation returns None."""
        attrs = empty_grid_db.get_substation_attributes("no_such_sub")
        assert attrs is None

    def test_partial_update_preserves_existing(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """Partial upsert preserves non-updated substation fields."""
        empty_grid_db.upsert_substation_attributes(
            "sub_1", voltage_setpoint_pu=1.02, zone="shikoku"
        )
        empty_grid_db.upsert_substation_attributes(
            "sub_1", tap_ratio=1.05
        )
        attrs = empty_grid_db.get_substation_attributes("sub_1")
        assert attrs.voltage_setpoint_pu == 1.02
        assert attrs.zone == "shikoku"
        assert attrs.tap_ratio == 1.05

    def test_delete_existing(self, empty_grid_db: GridDatabase) -> None:
        """Deleting an existing substation returns True."""
        empty_grid_db.upsert_substation_attributes("sub_1", zone="shikoku")
        result = empty_grid_db.delete_substation_attributes("sub_1")
        assert result is True
        assert empty_grid_db.get_substation_attributes("sub_1") is None

    def test_delete_nonexistent(self, empty_grid_db: GridDatabase) -> None:
        """Deleting a nonexistent substation returns False."""
        result = empty_grid_db.delete_substation_attributes("no_such_sub")
        assert result is False

    def test_list_substations(self, empty_grid_db: GridDatabase) -> None:
        """list_substation_attributes returns all stored records."""
        empty_grid_db.upsert_substation_attributes("sub_1", zone="shikoku")
        empty_grid_db.upsert_substation_attributes("sub_2", zone="chugoku")
        empty_grid_db.upsert_substation_attributes("sub_3", zone="kansai")
        records = empty_grid_db.list_substation_attributes()
        assert len(records) == 3

    def test_tap_range_fields(self, empty_grid_db: GridDatabase) -> None:
        """Tap range fields (tap_min, tap_max, tap_step_percent) stored."""
        empty_grid_db.upsert_substation_attributes(
            "sub_1",
            tap_min=0.9,
            tap_max=1.1,
            tap_step_percent=1.25,
        )
        attrs = empty_grid_db.get_substation_attributes("sub_1")
        assert attrs.tap_min == 0.9
        assert attrs.tap_max == 1.1
        assert attrs.tap_step_percent == 1.25


# ======================================================================
# GridDatabase: Load attributes CRUD
# ======================================================================


class TestLoadCRUD:
    """Tests for load attribute CRUD operations."""

    def test_create_and_read(self, empty_grid_db: GridDatabase) -> None:
        """Can store and retrieve load attributes."""
        empty_grid_db.upsert_load_attributes(
            "load_1",
            bus_id="bus_a",
            load_model="constant_power",
            p_mw=100.0,
            q_mvar=20.0,
            power_factor=0.95,
        )
        attrs = empty_grid_db.get_load_attributes("load_1")
        assert attrs is not None
        assert attrs.id == "load_1"
        assert attrs.bus_id == "bus_a"
        assert attrs.load_model == "constant_power"
        assert attrs.p_mw == 100.0
        assert attrs.q_mvar == 20.0
        assert attrs.power_factor == 0.95

    def test_read_nonexistent_returns_none(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """Reading a nonexistent load returns None."""
        attrs = empty_grid_db.get_load_attributes("no_such_load")
        assert attrs is None

    def test_partial_update_preserves_existing(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """Partial upsert preserves non-updated load fields."""
        empty_grid_db.upsert_load_attributes(
            "load_1", bus_id="bus_a", p_mw=100.0
        )
        empty_grid_db.upsert_load_attributes(
            "load_1", scaling_factor=0.85
        )
        attrs = empty_grid_db.get_load_attributes("load_1")
        assert attrs.bus_id == "bus_a"
        assert attrs.p_mw == 100.0
        assert attrs.scaling_factor == 0.85

    def test_delete_existing(self, empty_grid_db: GridDatabase) -> None:
        """Deleting an existing load returns True."""
        empty_grid_db.upsert_load_attributes("load_1", p_mw=50.0)
        result = empty_grid_db.delete_load_attributes("load_1")
        assert result is True
        assert empty_grid_db.get_load_attributes("load_1") is None

    def test_delete_nonexistent(self, empty_grid_db: GridDatabase) -> None:
        """Deleting a nonexistent load returns False."""
        result = empty_grid_db.delete_load_attributes("no_such_load")
        assert result is False

    def test_list_loads(self, empty_grid_db: GridDatabase) -> None:
        """list_load_attributes returns all stored records."""
        empty_grid_db.upsert_load_attributes("load_1", p_mw=100.0)
        empty_grid_db.upsert_load_attributes("load_2", p_mw=200.0)
        records = empty_grid_db.list_load_attributes()
        assert len(records) == 2

    def test_in_service_default(self, empty_grid_db: GridDatabase) -> None:
        """Load in_service defaults to 1."""
        empty_grid_db.upsert_load_attributes("load_1", p_mw=50.0)
        attrs = empty_grid_db.get_load_attributes("load_1")
        assert attrs.in_service == 1

    def test_source_field(self, empty_grid_db: GridDatabase) -> None:
        """Load source field is stored for traceability."""
        empty_grid_db.upsert_load_attributes(
            "load_1", source="synthetic_v1"
        )
        attrs = empty_grid_db.get_load_attributes("load_1")
        assert attrs.source == "synthetic_v1"


# ======================================================================
# GridDatabase: bulk operations
# ======================================================================


class TestBulkOperations:
    """Tests for bulk attribute retrieval."""

    def test_get_all_attributes_keys(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """get_all_attributes returns dict with expected keys."""
        result = empty_grid_db.get_all_attributes()
        assert "generators" in result
        assert "substations" in result
        assert "loads" in result

    def test_get_all_attributes_empty(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """get_all_attributes returns empty lists for empty DB."""
        result = empty_grid_db.get_all_attributes()
        assert result["generators"] == []
        assert result["substations"] == []
        assert result["loads"] == []

    def test_get_all_attributes_populated(
        self, sample_grid_db: GridDatabase
    ) -> None:
        """get_all_attributes returns populated lists from fixtures."""
        result = sample_grid_db.get_all_attributes()
        assert len(result["generators"]) == 2
        assert len(result["substations"]) == 2
        assert len(result["loads"]) == 1


# ======================================================================
# GridDatabase: pre-populated fixture validation
# ======================================================================


class TestSampleGridDB:
    """Tests verifying the sample_grid_db fixture data."""

    def test_coal_generator(self, sample_grid_db: GridDatabase) -> None:
        """Pre-populated coal generator has correct attributes."""
        attrs = sample_grid_db.get_generator_attributes("gen_coal_001")
        assert attrs is not None
        assert attrs.fuel_type == "coal"
        assert attrs.capacity_mw == 1460.0
        assert attrs.fuel_cost_per_mwh == 25.0
        assert attrs.startup_cost == 50000.0
        assert attrs.min_up_time_h == 8
        assert attrs.min_down_time_h == 6

    def test_nuclear_generator(self, sample_grid_db: GridDatabase) -> None:
        """Pre-populated nuclear generator has correct attributes."""
        attrs = sample_grid_db.get_generator_attributes("gen_nuclear_001")
        assert attrs is not None
        assert attrs.fuel_type == "nuclear"
        assert attrs.capacity_mw == 890.0
        assert attrs.fuel_cost_per_mwh == 8.0
        assert attrs.startup_cost == 200000.0
        assert attrs.min_up_time_h == 48

    def test_substations(self, sample_grid_db: GridDatabase) -> None:
        """Pre-populated substations have correct zone assignments."""
        sub1 = sample_grid_db.get_substation_attributes("sub_001")
        sub2 = sample_grid_db.get_substation_attributes("sub_002")
        assert sub1 is not None
        assert sub1.voltage_setpoint_pu == 1.02
        assert sub1.zone == "shikoku"
        assert sub2 is not None
        assert sub2.voltage_setpoint_pu == 1.00

    def test_load(self, sample_grid_db: GridDatabase) -> None:
        """Pre-populated load has correct attributes."""
        load = sample_grid_db.get_load_attributes("load_001")
        assert load is not None
        assert load.load_model == "constant_power"
        assert load.power_factor == 0.95
        assert load.scaling_factor == 1.0


# ======================================================================
# Schema versioning
# ======================================================================


class TestSchemaVersioning:
    """Tests for schema version tracking."""

    def test_initial_version_is_one(self) -> None:
        """Fresh in-memory database starts at schema version 1."""
        db = GridDatabase(":memory:")
        assert db.get_schema_version() == 1

    def test_version_persists_across_sessions(self, tmp_path) -> None:
        """Schema version persists when database is reopened."""
        db_path = str(tmp_path / "version_test.db")
        db1 = GridDatabase(db_path)
        v1 = db1.get_schema_version()

        db2 = GridDatabase(db_path)
        v2 = db2.get_schema_version()
        assert v2 == v1

    def test_schema_version_table_populated(self) -> None:
        """schema_version table has at least one row after init."""
        engine = create_engine("sqlite:///:memory:")
        manager = MigrationManager(engine)
        manager.ensure_schema()

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM schema_version")
            )
            count = result.scalar()
            assert count >= 1


# ======================================================================
# Migration manager
# ======================================================================


class TestMigrationManager:
    """Tests for the lightweight migration system."""

    def test_get_current_version_empty_db(self) -> None:
        """Current version is 0 for a DB with no schema_version table."""
        engine = create_engine("sqlite:///:memory:")
        manager = MigrationManager(engine)
        assert manager.get_current_version() == 0

    def test_ensure_schema_creates_tables(self) -> None:
        """ensure_schema creates all ORM-defined tables."""
        engine = create_engine("sqlite:///:memory:")
        manager = MigrationManager(engine)
        manager.ensure_schema()

        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "generator_attributes" in tables
        assert "substation_attributes" in tables
        assert "load_attributes" in tables
        assert "schema_version" in tables

    def test_ensure_schema_returns_version(self) -> None:
        """ensure_schema returns the current schema version."""
        engine = create_engine("sqlite:///:memory:")
        manager = MigrationManager(engine)
        version = manager.ensure_schema()
        assert version >= 1

    def test_ensure_schema_idempotent(self) -> None:
        """Calling ensure_schema twice does not change the version."""
        engine = create_engine("sqlite:///:memory:")
        manager = MigrationManager(engine)
        v1 = manager.ensure_schema()
        v2 = manager.ensure_schema()
        assert v1 == v2

    def test_apply_pending_no_pending(self) -> None:
        """apply_pending returns empty list when up-to-date."""
        engine = create_engine("sqlite:///:memory:")
        manager = MigrationManager(engine)
        manager.ensure_schema()
        applied = manager.apply_pending()
        assert applied == []

    def test_custom_migration_applied(self) -> None:
        """A custom migration with ALTER TABLE is applied correctly."""
        engine = create_engine("sqlite:///:memory:")
        manager = MigrationManager(engine)
        manager.ensure_schema()

        # Manually apply a v2 migration that adds a column
        migration = Migration(
            version=2,
            description="Add test_column to generator_attributes",
            statements=[
                "ALTER TABLE generator_attributes ADD COLUMN test_column TEXT"
            ],
        )
        manager._apply(migration)

        # Verify the column was added
        columns = manager.get_table_columns("generator_attributes")
        assert "test_column" in columns

        # Verify version was recorded
        assert manager.get_current_version() == 2

    def test_migration_preserves_existing_data(self) -> None:
        """Schema migration adds columns without losing existing data."""
        engine = create_engine("sqlite:///:memory:")
        manager = MigrationManager(engine)
        manager.ensure_schema()

        # Insert data before migration
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO generator_attributes (id, fuel_type, capacity_mw) "
                    "VALUES ('gen_pre', 'coal', 500.0)"
                )
            )

        # Apply migration adding a new column
        migration = Migration(
            version=2,
            description="Add emission_factor to generator_attributes",
            statements=[
                "ALTER TABLE generator_attributes ADD COLUMN emission_factor REAL"
            ],
        )
        manager._apply(migration)

        # Verify existing data is preserved
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT fuel_type, capacity_mw FROM generator_attributes WHERE id = 'gen_pre'")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == "coal"
            assert row[1] == 500.0

    def test_get_table_columns(self) -> None:
        """get_table_columns returns column names for existing table."""
        engine = create_engine("sqlite:///:memory:")
        manager = MigrationManager(engine)
        manager.ensure_schema()

        columns = manager.get_table_columns("generator_attributes")
        assert "id" in columns
        assert "fuel_type" in columns
        assert "capacity_mw" in columns

    def test_get_table_columns_nonexistent(self) -> None:
        """get_table_columns returns empty list for nonexistent table."""
        engine = create_engine("sqlite:///:memory:")
        manager = MigrationManager(engine)
        columns = manager.get_table_columns("nonexistent_table")
        assert columns == []

    def test_registered_migrations_sequential(self) -> None:
        """Registered MIGRATIONS have sequential version numbers."""
        versions = [m.version for m in MIGRATIONS]
        assert versions == sorted(versions)
        # No duplicates
        assert len(versions) == len(set(versions))

    def test_migration_version_one_exists(self) -> None:
        """MIGRATIONS contains at least version 1 (initial schema)."""
        versions = [m.version for m in MIGRATIONS]
        assert 1 in versions


# ======================================================================
# SQLite pragmas
# ======================================================================


class TestSQLitePragmas:
    """Tests for SQLite pragma configuration (WAL, foreign keys)."""

    def test_wal_mode_enabled(self, tmp_path) -> None:
        """WAL journal mode is enabled for file-based databases."""
        db_path = str(tmp_path / "wal_test.db")
        db = GridDatabase(db_path)
        # Access the internal engine to check pragma
        with db._engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode"))
            mode = result.scalar()
            assert mode == "wal"

    def test_foreign_keys_enabled(self) -> None:
        """Foreign keys pragma is enabled."""
        db = GridDatabase(":memory:")
        with db._engine.connect() as conn:
            result = conn.execute(text("PRAGMA foreign_keys"))
            fk = result.scalar()
            assert fk == 1


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_upsert_with_no_kwargs_creates_minimal_record(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """Upsert with only ID creates a record with null fields."""
        empty_grid_db.upsert_generator_attributes("gen_minimal")
        attrs = empty_grid_db.get_generator_attributes("gen_minimal")
        assert attrs is not None
        assert attrs.id == "gen_minimal"
        assert attrs.fuel_type is None
        assert attrs.capacity_mw is None

    def test_upsert_same_id_multiple_times(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """Multiple upserts to the same ID accumulate fields."""
        empty_grid_db.upsert_generator_attributes("gen_1", fuel_type="coal")
        empty_grid_db.upsert_generator_attributes("gen_1", capacity_mw=500.0)
        empty_grid_db.upsert_generator_attributes("gen_1", vm_pu=1.02)

        attrs = empty_grid_db.get_generator_attributes("gen_1")
        assert attrs.fuel_type == "coal"
        assert attrs.capacity_mw == 500.0
        assert attrs.vm_pu == 1.02

    def test_delete_then_recreate(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """Deleted record can be recreated with new attributes."""
        empty_grid_db.upsert_generator_attributes(
            "gen_1", fuel_type="coal"
        )
        empty_grid_db.delete_generator_attributes("gen_1")
        empty_grid_db.upsert_generator_attributes(
            "gen_1", fuel_type="lng", capacity_mw=800.0
        )
        attrs = empty_grid_db.get_generator_attributes("gen_1")
        assert attrs.fuel_type == "lng"
        assert attrs.capacity_mw == 800.0

    def test_many_records(self, empty_grid_db: GridDatabase) -> None:
        """Database handles many records without error."""
        for i in range(100):
            empty_grid_db.upsert_generator_attributes(
                f"gen_{i:03d}", capacity_mw=float(i * 10)
            )
        records = empty_grid_db.list_generator_attributes()
        assert len(records) == 100

    def test_unicode_values(self, empty_grid_db: GridDatabase) -> None:
        """Unicode strings are stored and retrieved correctly."""
        empty_grid_db.upsert_substation_attributes(
            "sub_jp", zone="四国", grid_class="幹線系統"
        )
        attrs = empty_grid_db.get_substation_attributes("sub_jp")
        assert attrs.zone == "四国"
        assert attrs.grid_class == "幹線系統"

    def test_cross_table_independence(
        self, empty_grid_db: GridDatabase
    ) -> None:
        """Operations on different tables are independent."""
        empty_grid_db.upsert_generator_attributes("id_1", fuel_type="coal")
        empty_grid_db.upsert_substation_attributes("id_1", zone="test")
        empty_grid_db.upsert_load_attributes("id_1", p_mw=50.0)

        gen = empty_grid_db.get_generator_attributes("id_1")
        sub = empty_grid_db.get_substation_attributes("id_1")
        load = empty_grid_db.get_load_attributes("id_1")

        assert gen.fuel_type == "coal"
        assert sub.zone == "test"
        assert load.p_mw == 50.0

        # Deleting from one table does not affect others
        empty_grid_db.delete_generator_attributes("id_1")
        assert empty_grid_db.get_generator_attributes("id_1") is None
        assert empty_grid_db.get_substation_attributes("id_1") is not None
        assert empty_grid_db.get_load_attributes("id_1") is not None
