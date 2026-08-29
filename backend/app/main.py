"""
FastAPI application entrypoint.

Phase 1 scope: app bootstrapping, structured logging setup, and a health
check only. Routers for /api/flows, /api/alerts, /api/stats, /api/config,
and the /ws/alerts + /ws/metrics WebSocket endpoints are added in
Phases 11-12 once their underlying components exist.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("nids")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("NIDS backend starting in '%s' environment", settings.environment)
    logger.info("Configured capture interface: %s", settings.capture_interface)
    yield
    logger.info("NIDS backend shutting down")


app = FastAPI(
    title="Real-Time Network Intrusion Detection System",
    description="Hybrid rule-based + ML network intrusion detection for an authorized lab environment.",
    version="0.1.0",
    lifespan=lifespan,
)

# Dashboard dev server (Vite) origin -- tighten this before any non-lab deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Basic liveness check. Extended in later phases with DB/capture status."""
    return {"status": "ok", "environment": settings.environment}
