"""Database layer for schema-versioned grid attribute storage.

Provides SQLAlchemy 2.0+ ORM models and CRUD operations for mutable
generator, substation, and load attributes.  Uses a SQLite backend for
zero-configuration operation — no external database server required.

Usage::

    from src.db.schema import Base, GeneratorAttributes, SubstationAttributes
    from src.db.grid_db import GridDatabase

    db = GridDatabase("data/grid_attributes.db")
    db.upsert_generator_attributes("gen_1", fuel_type="coal", capacity_mw=500.0)
    attrs = db.get_generator_attributes("gen_1")
"""
