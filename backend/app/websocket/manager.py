"""
Phase 10: WebSocket connection management for real-time detection events.

    HybridDetectionResult -> ConnectionManager.broadcast() -> all connected clients

No orchestrator continuously feeds live detections into this yet -- same
status as Phases 7-9's ready-but-not-wired-into-a-live-loop capabilities
(see docs/architecture.md). This module provides the connection lifecycle
and broadcast mechanism, fully tested via FastAPI's WebSocket test client,
ready for whichever future phase adds a live capture-to-detection service
loop that calls `connection_manager.broadcast(...)` per new detection.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket

logger = logging.getLogger("nids.websocket")


class ConnectionManager:
    """
    Tracks connected WebSocket clients and broadcasts JSON messages to all
    of them, safely handling individual client disconnects/failures
    without affecting other clients or raising out of `broadcast()`.
    """

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info("WebSocket client connected (%d total).", len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
        logger.info("WebSocket client disconnected (%d total).", len(self._connections))

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    async def broadcast(self, message: dict) -> None:
        """
        Send `message` as JSON to every connected client. A failure
        sending to any one client (e.g. it dropped without a clean
        WebSocket close handshake) is caught and that connection is
        removed -- it never prevents delivery to the other clients, and
        `broadcast()` itself never raises because one client misbehaved.
        """
        async with self._lock:
            targets = list(self._connections)

        dead: list[WebSocket] = []
        for connection in targets:
            try:
                await connection.send_json(message)
            except Exception:
                logger.warning("Failed to send to a WebSocket client -- dropping it.", exc_info=True)
                dead.append(connection)

        if dead:
            async with self._lock:
                for connection in dead:
                    self._connections.discard(connection)


# Process-wide singleton. The /ws/alerts endpoint registers/unregisters
# connections here; a future live-processing loop (or this phase's own
# tests) calls `connection_manager.broadcast(...)` to push new detection
# events to every connected client.
connection_manager = ConnectionManager()
