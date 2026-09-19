"""
CLI entry point for Phase 9 database schema initialization.

Usage:
    python -m app.database.init_db_cli

Creates the `flows` and `detections` tables (if they don't already exist)
at `settings.database_url` (i.e. `DATABASE_URL` from `.env`). See
docker-compose.yml for a local PostgreSQL instance matching the default
connection string, and README.md for full setup instructions.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import settings
from app.database.session import create_db_engine, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("nids.database.init_db_cli")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialize the NIDS PostgreSQL schema (flows + detections tables).")
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override the database URL (defaults to settings.database_url / the DATABASE_URL env var).",
    )
    args = parser.parse_args(argv)

    database_url = args.database_url or settings.database_url
    logger.info("Initializing schema at %s", database_url)

    try:
        engine = create_db_engine(database_url)
        init_db(engine)
    except Exception as exc:  # noqa: BLE001 -- top-level CLI: report and exit non-zero rather than a raw traceback
        logger.error("Schema initialization failed: %s", exc)
        return 1

    print(f"Schema initialized successfully at {database_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
