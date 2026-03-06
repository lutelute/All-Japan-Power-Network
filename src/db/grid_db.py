"""Grid attribute database with CRUD operations and schema versioning.

Wraps SQLAlchemy session management and provides a clean API for
creating, reading, updating, and deleting generator, substation, and
load attributes.  Uses the upsert pattern (create-or-update) so callers
can set partial attributes without overwriting previously stored values.

The database auto-migrates on initialisation — schema changes registered
in :mod:`src.db.migrations` are applied transparently.

Usage::

    from src.db.grid_db import GridDatabase

    db = GridDatabase("data/grid_attributes.db")

    # Upsert generator attributes (partial update supported)
    db.upsert_generator_attributes("gen_1", fuel_type="coal", capacity_mw=500.0)

    # Later, add cost data without losing existing attributes
    db.upsert_generator_attributes("gen_1", fuel_cost_per_mwh=25.0)

    attrs = db.get_generator_attributes("gen_1")
    # attrs.fuel_type == "coal", attrs.fuel_cost_per_mwh == 25.0

    # In-memory database for testing
    db = GridDatabase(":memory:")
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Type, TypeVar

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.db.migrations import MigrationManager
from src.db.schema import (
    Base,
    GeneratorAttributes,
    LoadAttributes,
    SchemaVersion,
    SubstationAttributes,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=Base)


def _set_sqlite_pragmas(dbapi_conn, connection_record) -> None:  # noqa: ANN001
    """Enable WAL mode and foreign keys for each SQLite connection.

    Registered as a ``connect`` event listener so that every new raw
    DBAPI connection gets the pragmas applied automatically.
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class GridDatabase:
    """Grid attribute database with CRUD operations and auto-migration.

    Creates (or connects to) a SQLite database at the given path,
    applies any pending schema migrations, and exposes typed CRUD
    methods for generator, substation, and load attribute records.

    Args:
        db_path: Path to the SQLite database file, or ``':memory:'``
            for an in-memory database (useful for testing).
        echo: If ``True``, emit SQL statements to the logger for
            debugging.  Defaults to ``False``.
    """

    def __init__(self, db_path: str, echo: bool = False) -> None:
        if db_path == ":memory:":
            url = "sqlite:///:memory:"
        else:
            url = f"sqlite:///{db_path}"

        self._engine: Engine = create_engine(url, echo=echo)

        # Enable WAL mode and foreign keys for SQLite
        event.listen(self._engine, "connect", _set_sqlite_pragmas)

        # Auto-migrate: create tables and apply pending migrations
        self._migration_manager = MigrationManager(self._engine)
        self._migration_manager.ensure_schema()

        self._session_factory = sessionmaker(bind=self._engine)

        logger.info(
            "GridDatabase initialised (path=%s, schema_version=%d)",
            db_path,
            self.get_schema_version(),
        )

    # ------------------------------------------------------------------
    # Schema version
    # ------------------------------------------------------------------

    def get_schema_version(self) -> int:
        """Return the current database schema version.

        Returns:
            The highest applied migration version number.
        """
        return self._migration_manager.get_current_version()

    # ------------------------------------------------------------------
    # Generator attributes CRUD
    # ------------------------------------------------------------------

    def upsert_generator_attributes(
        self, gen_id: str, **kwargs: object
    ) -> GeneratorAttributes:
        """Create or update generator attributes.

        If a record with ``gen_id`` already exists, only the provided
        keyword arguments are updated — existing fields are preserved.
        This enables partial attribute updates without data loss.

        Args:
            gen_id: Generator identifier.
            **kwargs: Attribute names and values to set.  Must match
                column names on :class:`~src.db.schema.GeneratorAttributes`.

        Returns:
            The created or updated :class:`GeneratorAttributes` instance.

        Raises:
            AttributeError: If an invalid attribute name is provided.
        """
        return self._upsert(GeneratorAttributes, gen_id, **kwargs)

    def get_generator_attributes(
        self, gen_id: str
    ) -> Optional[GeneratorAttributes]:
        """Retrieve generator attributes by ID.

        Args:
            gen_id: Generator identifier.

        Returns:
            The :class:`GeneratorAttributes` record, or ``None`` if not found.
        """
        return self._get(GeneratorAttributes, gen_id)

    def list_generator_attributes(self) -> List[GeneratorAttributes]:
        """Return all generator attribute records.

        Returns:
            List of :class:`GeneratorAttributes` instances.
        """
        return self._list_all(GeneratorAttributes)

    def delete_generator_attributes(self, gen_id: str) -> bool:
        """Delete a generator attribute record.

        Args:
            gen_id: Generator identifier.

        Returns:
            ``True`` if the record existed and was deleted, ``False``
            if no record was found.
        """
        return self._delete(GeneratorAttributes, gen_id)

    # ------------------------------------------------------------------
    # Substation attributes CRUD
    # ------------------------------------------------------------------

    def upsert_substation_attributes(
        self, sub_id: str, **kwargs: object
    ) -> SubstationAttributes:
        """Create or update substation attributes.

        If a record with ``sub_id`` already exists, only the provided
        keyword arguments are updated — existing fields are preserved.

        Args:
            sub_id: Substation identifier.
            **kwargs: Attribute names and values to set.

        Returns:
            The created or updated :class:`SubstationAttributes` instance.
        """
        return self._upsert(SubstationAttributes, sub_id, **kwargs)

    def get_substation_attributes(
        self, sub_id: str
    ) -> Optional[SubstationAttributes]:
        """Retrieve substation attributes by ID.

        Args:
            sub_id: Substation identifier.

        Returns:
            The :class:`SubstationAttributes` record, or ``None`` if not found.
        """
        return self._get(SubstationAttributes, sub_id)

    def list_substation_attributes(self) -> List[SubstationAttributes]:
        """Return all substation attribute records.

        Returns:
            List of :class:`SubstationAttributes` instances.
        """
        return self._list_all(SubstationAttributes)

    def delete_substation_attributes(self, sub_id: str) -> bool:
        """Delete a substation attribute record.

        Args:
            sub_id: Substation identifier.

        Returns:
            ``True`` if the record existed and was deleted, ``False``
            if no record was found.
        """
        return self._delete(SubstationAttributes, sub_id)

    # ------------------------------------------------------------------
    # Load attributes CRUD
    # ------------------------------------------------------------------

    def upsert_load_attributes(
        self, load_id: str, **kwargs: object
    ) -> LoadAttributes:
        """Create or update load attributes.

        If a record with ``load_id`` already exists, only the provided
        keyword arguments are updated — existing fields are preserved.

        Args:
            load_id: Load identifier.
            **kwargs: Attribute names and values to set.

        Returns:
            The created or updated :class:`LoadAttributes` instance.
        """
        return self._upsert(LoadAttributes, load_id, **kwargs)

    def get_load_attributes(self, load_id: str) -> Optional[LoadAttributes]:
        """Retrieve load attributes by ID.

        Args:
            load_id: Load identifier.

        Returns:
            The :class:`LoadAttributes` record, or ``None`` if not found.
        """
        return self._get(LoadAttributes, load_id)

    def list_load_attributes(self) -> List[LoadAttributes]:
        """Return all load attribute records.

        Returns:
            List of :class:`LoadAttributes` instances.
        """
        return self._list_all(LoadAttributes)

    def delete_load_attributes(self, load_id: str) -> bool:
        """Delete a load attribute record.

        Args:
            load_id: Load identifier.

        Returns:
            ``True`` if the record existed and was deleted, ``False``
            if no record was found.
        """
        return self._delete(LoadAttributes, load_id)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def get_all_attributes(self) -> Dict[str, List[Base]]:
        """Return all attribute records grouped by table.

        Returns:
            Dict with keys ``'generators'``, ``'substations'``,
            ``'loads'`` mapping to their respective attribute lists.
        """
        return {
            "generators": self.list_generator_attributes(),
            "substations": self.list_substation_attributes(),
            "loads": self.list_load_attributes(),
        }

    # ------------------------------------------------------------------
    # Internal generic CRUD helpers
    # ------------------------------------------------------------------

    def _upsert(self, model: Type[T], record_id: str, **kwargs: object) -> T:
        """Create or update a record with partial attribute support.

        If the record already exists, only the provided kwargs are
        updated.  Existing attribute values that are not in kwargs are
        preserved.  The ``updated_at`` timestamp is refreshed
        automatically.

        Args:
            model: The SQLAlchemy ORM model class.
            record_id: Primary key value.
            **kwargs: Attribute names and values to set.

        Returns:
            The created or updated ORM instance (detached from session).
        """
        with self._session_factory() as session:
            record = session.get(model, record_id)

            if record is None:
                # Create new record
                record = model(id=record_id, **kwargs)
                if hasattr(record, "updated_at"):
                    record.updated_at = datetime.now(timezone.utc)
                session.add(record)
            else:
                # Update only provided attributes
                for attr_name, attr_value in kwargs.items():
                    if not hasattr(record, attr_name):
                        raise AttributeError(
                            f"{model.__name__} has no attribute '{attr_name}'"
                        )
                    setattr(record, attr_name, attr_value)
                if hasattr(record, "updated_at"):
                    record.updated_at = datetime.now(timezone.utc)

            session.commit()

            # Detach and return — expunge to avoid lazy-load issues
            session.expunge(record)
            return record

    def _get(self, model: Type[T], record_id: str) -> Optional[T]:
        """Retrieve a single record by primary key.

        Args:
            model: The SQLAlchemy ORM model class.
            record_id: Primary key value.

        Returns:
            The ORM instance, or ``None`` if not found.
        """
        with self._session_factory() as session:
            record = session.get(model, record_id)
            if record is not None:
                session.expunge(record)
            return record

    def _list_all(self, model: Type[T]) -> List[T]:
        """Retrieve all records for a given model.

        Args:
            model: The SQLAlchemy ORM model class.

        Returns:
            List of all ORM instances for the table.
        """
        with self._session_factory() as session:
            records = list(session.query(model).all())
            for record in records:
                session.expunge(record)
            return records

    def _delete(self, model: Type[T], record_id: str) -> bool:
        """Delete a record by primary key.

        Args:
            model: The SQLAlchemy ORM model class.
            record_id: Primary key value.

        Returns:
            ``True`` if the record was found and deleted, ``False`` otherwise.
        """
        with self._session_factory() as session:
            record = session.get(model, record_id)
            if record is None:
                return False
            session.delete(record)
            session.commit()
            return True
