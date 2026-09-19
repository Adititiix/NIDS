"""
Phase 10: shared FastAPI dependencies.

One engine/session factory is created at import time and reused across
requests, following SQLAlchemy's own recommended usage (an Engine is meant
to be a long-lived, process-wide object, not recreated per request).
Creating an Engine does not open a connection eagerly -- it's lazy, so
importing this module is safe even without a reachable database (a
request will only fail once it actually needs the database, and that
failure is handled per-route via DatabasePersistenceError -> HTTP 503).

Tests override `get_repository` via FastAPI's `app.dependency_overrides`
rather than relying on this module's real database_url -- see
tests/test_api.py.
"""

from __future__ import annotations

from app.config import settings
from app.database.repository import FlowDetectionRepository
from app.database.session import create_db_engine, create_session_factory

_engine = create_db_engine(settings.database_url)
_session_factory = create_session_factory(_engine)


def get_repository() -> FlowDetectionRepository:
    """FastAPI dependency: a repository bound to the shared, process-wide engine/session factory."""
    return FlowDetectionRepository(_session_factory)
