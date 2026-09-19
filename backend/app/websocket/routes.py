"""Phase 10: WS /ws/alerts -- real-time detection event stream."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.websocket.manager import connection_manager

router = APIRouter()


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket) -> None:
    """
    Clients connect and receive a JSON message for every new detection
    event broadcast via `connection_manager.broadcast(...)`. This endpoint
    doesn't expect meaningful messages FROM the client -- it's a
    server-push feed -- but still awaits `receive_text()` in a loop
    purely to detect a disconnect promptly (rather than polling), and
    unregisters the connection in a `finally` block so a client that
    disconnects uncleanly can never leave a stale/blocking entry behind.
    """
    await connection_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.disconnect(websocket)
