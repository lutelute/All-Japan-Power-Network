"""Lightweight schema migration system for the grid attribute database.

Provides automatic schema versioning and migration using SQLite
``ALTER TABLE ADD COLUMN`` statements.  No external migration framework
(e.g. Alembic) is required — all logic is self-contained for minimal
dependencies.

Migration workflow:

1. On :class:`MigrationManager` initialisation the ``schema_version``
   table is inspected (or created) to determine the current version.
2. :meth:`apply_pending` iterates over registered migrations whose
   version number exceeds the current database version and applies
   them in order.
3. Each migration records its version in the ``schema_version`` table
   so it will not be re-applied on subsequent runs.

Adding a new migration:

    Append a :class:`Migration` to :data:`MIGRATIONS` with the next
    sequential version number and the ``ALTER TABLE ADD COLUMN``
    statements needed.  Existing data is preserved because ``ADD COLUMN``
    never drops data.

Usage::

    from sqlalchemy import create_engine
    from src.db.migrations import MigrationManager

    engine = create_engine("sqlite:///data/grid_attributes.db")
    manager = MigrationManager(engine)
    applied = manager.apply_pending()
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Sequence

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Migration:
    """A single schema migration step.

    Attributes:
        version: Monotonically increasing version number (must be >= 1).
        description: Human-readable description of the migration.
        statements: SQL statements to execute for this migration.
    """

    version: int
    description: str
    statements: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registered migrations — append new entries at the end.
# ---------------------------------------------------------------------------

MIGRATIONS: Sequence[Migration] = (
    Migration(
        version=1,
        description="Initial schema — tables created by SQLAlchemy metadata",
        statements=[],
    ),
)


class MigrationManager:
    """Manages lightweight schema migrations for the grid attribute DB.

    Ensures the ``schema_version`` tracking table exists, creates all
    ORM-defined tables via ``Base.metadata.create_all()``, and applies
    any registered :class:`Migration` steps whose version exceeds the
    current database version.

    Args:
        engine: SQLAlchemy engine connected to the target database.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_version(self) -> int:
        """Return the highest applied schema version, or 0 if none."""
        inspector = inspect(self._engine)
        if "schema_version" not in inspector.get_table_names():
            return 0

        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT MAX(version) FROM schema_version"
                )
            )
            row = result.scalar()
            return row if row is not None else 0

    def apply_pending(self) -> List[int]:
        """Apply all migrations whose version exceeds the current state.

        Returns:
            List of version numbers that were applied (empty if
            already up-to-date).
        """
        current = self.get_current_version()
        pending = [m for m in MIGRATIONS if m.version > current]

        if not pending:
            logger.debug("Schema is up-to-date at version %d", current)
            return []

        applied: List[int] = []

        for migration in sorted(pending, key=lambda m: m.version):
            self._apply(migration)
            applied.append(migration.version)

        logger.info(
            "Applied %d migration(s): versions %s (now at v%d)",
            len(applied),
            applied,
            applied[-1],
        )
        return applied

    def ensure_schema(self) -> int:
        """Create tables if needed and apply pending migrations.

        This is the primary entry point called during
        :class:`~src.db.grid_db.GridDatabase` initialisation.

        Returns:
            The current schema version after all migrations.
        """
        from src.db.schema import Base

        # Create any missing tables (idempotent)
        Base.metadata.create_all(self._engine)

        # Seed initial version if schema_version table is empty
        current = self.get_current_version()
        if current == 0:
            self._record_version(
                version=1,
                description="Initial schema — tables created by SQLAlchemy metadata",
            )
            current = 1

        # Apply subsequent migrations
        self.apply_pending()

        return self.get_current_version()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply(self, migration: Migration) -> None:
        """Execute a single migration's SQL statements and record it.

        Args:
            migration: The migration to apply.
        """
        logger.info(
            "Applying migration v%d: %s",
            migration.version,
            migration.description,
        )

        with self._engine.begin() as conn:
            for stmt in migration.statements:
                try:
                    conn.execute(text(stmt))
                except Exception:
                    logger.exception(
                        "Failed to execute migration v%d statement: %s",
                        migration.version,
                        stmt,
                    )
                    raise

        self._record_version(
            version=migration.version,
            description=migration.description,
        )

    def _record_version(self, version: int, description: str) -> None:
        """Insert a row into ``schema_version`` for a completed migration.

        Args:
            version: The migration version number.
            description: Human-readable description of the migration.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT OR REPLACE INTO schema_version "
                    "(version, description, applied_at) "
                    "VALUES (:version, :description, :applied_at)"
                ),
                {
                    "version": version,
                    "description": description,
                    "applied_at": now,
                },
            )

    def get_table_columns(self, table_name: str) -> List[str]:
        """Return column names for the given table.

        Useful for migration logic that needs to check which columns
        already exist before adding new ones.

        Args:
            table_name: The database table to inspect.

        Returns:
            List of column name strings.
        """
        inspector = inspect(self._engine)
        if table_name not in inspector.get_table_names():
            return []
        return [col["name"] for col in inspector.get_columns(table_name)]
