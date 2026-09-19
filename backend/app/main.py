"""
FastAPI application entrypoint.

Phase 1 scope was app bootstrapping, structured logging, and a health
check only. Phase 10 adds the REST API routers (/api/flows, /api/alerts,
/api/stats, /api/status) and the /ws/alerts WebSocket endpoint. The
original /api/health liveness check below is unchanged from Phase 1.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import alerts, flows, stats, system
from app.config import settings
from app.websocket import routes as websocket_routes

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


app.include_router(system.router)
app.include_router(flows.router)
app.include_router(alerts.router)
app.include_router(stats.router)
app.include_router(websocket_routes.router)


@app.get("/api/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Basic liveness check. Extended in later phases with DB/capture status."""
    return {"status": "ok", "environment": settings.environment}
