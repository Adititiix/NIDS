"""
Phase 9: database engine and session management.

Connection configuration comes entirely from `app.config.settings.database_url`
(populated from the `DATABASE_URL` environment variable, per `.env.example`)
-- no credentials or connection strings are hardcoded here.
"""

from __future__ import annotations

import logging

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.database.models import Base

logger = logging.getLogger("nids.database")


def create_db_engine(database_url: str | None = None, **engine_kwargs) -> Engine:
    """
    Create a SQLAlchemy engine for `database_url` (defaults to
    `settings.database_url`, i.e. `DATABASE_URL` from `.env`).

    For SQLite engines specifically (used by this project's own test
    suite, never in production), foreign-key enforcement is turned on
    explicitly -- SQLite has it off by default, which would silently let
    an invalid `flow_id` be inserted into `detections` and defeat the
    graceful-failure test coverage this phase requires. PostgreSQL enforces
    foreign keys unconditionally, so this only matters for SQLite.
    """
    url = database_url or settings.database_url
    engine = create_engine(url, **engine_kwargs)

    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """
    `expire_on_commit=False` so a record's attributes (e.g. a newly
    generated `id`) remain readable immediately after `session.commit()`
    without a fresh round-trip -- the repository layer still constructs
    its own plain dataclasses before returning, so this is a minor
    convenience, not something callers should rely on for reads.
    """
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """
    Create all tables defined in app.database.models if they don't already
    exist.

    This is a deliberately lightweight schema-initialization mechanism,
    sufficient for this project's current stage (one schema version, no
    migrations to apply yet). The repo's `database/migrations/` directory
    is reserved for a dedicated migration tool (Alembic is the natural
    choice, given SQLAlchemy is already in use) once the schema needs to
    evolve across versions in a running deployment -- introducing that
    tooling now, before there is a second schema version to migrate
    between, would be complexity without a corresponding need. See
    docs/methodology.md for this project's full Phase 9 writeup.
    """
    logger.info("Initializing database schema (create_all, no-op for existing tables)...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ready.")
