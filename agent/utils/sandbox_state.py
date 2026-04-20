"""Shared sandbox state used by server and middleware."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from langgraph.config import get_config

from .sandbox import create_sandbox

logger = logging.getLogger(__name__)

_STATE_DB_PATH = Path(os.environ.get("OPEN_SWE_STATE_DB", Path(__file__).resolve().parents[4] / ".open_swe_acp_sessions.sqlite3"))
_STATE_DB_LOCK = threading.Lock()


def _ensure_state_db() -> None:
    _STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_STATE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS acp_sessions (
                session_id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                mcp_servers_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def _db_execute(query: str, params: tuple[Any, ...] = ()) -> None:
    with _STATE_DB_LOCK:
        _ensure_state_db()
        with sqlite3.connect(_STATE_DB_PATH) as conn:
            conn.execute(query, params)
            conn.commit()


def _db_fetchone(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with _STATE_DB_LOCK:
        _ensure_state_db()
        with sqlite3.connect(_STATE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(query, params).fetchone()
            return dict(row) if row else None


def _db_fetchall(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with _STATE_DB_LOCK:
        _ensure_state_db()
        with sqlite3.connect(_STATE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]


def _serialize_mcp_server(server: Any) -> dict[str, Any]:
    if isinstance(server, dict):
        return server
    payload: dict[str, Any] = {}
    for key in ("name", "command", "args", "url", "headers", "env"):
        value = getattr(server, key, None)
        if value is not None:
            if key == "env" and isinstance(value, list):
                payload[key] = [
                    {"name": getattr(item, "name", None), "value": getattr(item, "value", None)}
                    for item in value
                ]
            else:
                payload[key] = value
    payload["type"] = type(server).__name__
    return payload


def persist_acp_session(session_id: str, cwd: str, mcp_servers: list[Any] | None = None) -> None:
    payload = {
        "cwd": cwd,
        "metadata": {},
        "mcp_servers": [_serialize_mcp_server(server) for server in (mcp_servers or [])],
    }
    _db_execute(
        """
        INSERT INTO acp_sessions (session_id, cwd, metadata_json, mcp_servers_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            cwd = excluded.cwd,
            mcp_servers_json = excluded.mcp_servers_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (session_id, payload["cwd"], json.dumps(payload["metadata"]), json.dumps(payload["mcp_servers"])),
    )


def persist_acp_session_metadata(session_id: str, metadata: dict[str, Any]) -> None:
    current = get_thread_metadata(session_id)
    merged = {**current, **metadata}
    _db_execute(
        """
        UPDATE acp_sessions
        SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP
        WHERE session_id = ?
        """,
        (json.dumps(merged), session_id),
    )


def list_acp_sessions() -> list[dict[str, Any]]:
    rows = _db_fetchall(
        """
        SELECT session_id, cwd, metadata_json, mcp_servers_json, created_at, updated_at
        FROM acp_sessions
        ORDER BY updated_at DESC
        """
    )
    sessions: list[dict[str, Any]] = []
    for row in rows:
        sessions.append(
            {
                "session_id": row["session_id"],
                "cwd": row["cwd"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "mcp_servers": json.loads(row["mcp_servers_json"] or "[]"),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    return sessions


def get_acp_session(session_id: str) -> dict[str, Any] | None:
    row = _db_fetchone(
        """
        SELECT session_id, cwd, metadata_json, mcp_servers_json, created_at, updated_at
        FROM acp_sessions
        WHERE session_id = ?
        """,
        (session_id,),
    )
    if not row:
        return None
    return {
        "session_id": row["session_id"],
        "cwd": row["cwd"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
        "mcp_servers": json.loads(row["mcp_servers_json"] or "[]"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }

# Thread ID -> SandboxBackend mapping, shared between server.py and middleware
SANDBOX_BACKENDS: dict[str, Any] = {}
# Thread ID -> protocol metadata for ACP-local sessions
THREAD_METADATA: dict[str, dict[str, Any]] = {}


def get_thread_metadata(thread_id: str) -> dict[str, Any]:
    cached = THREAD_METADATA.get(thread_id)
    if cached is not None:
        return cached
    row = get_acp_session(thread_id)
    if not row:
        return {}
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        THREAD_METADATA[thread_id] = metadata
        return metadata
    return {}


def update_thread_metadata(thread_id: str, metadata: dict[str, Any]) -> None:
    current = THREAD_METADATA.get(thread_id, {})
    merged = {**current, **metadata}
    THREAD_METADATA[thread_id] = merged
    persist_acp_session_metadata(thread_id, merged)


async def get_sandbox_id_from_metadata(thread_id: str) -> str | None:
    """Fetch sandbox_id from thread metadata."""
    metadata = get_thread_metadata(thread_id)
    sandbox_id = metadata.get("sandbox_id")
    if isinstance(sandbox_id, str) and sandbox_id:
        return sandbox_id
    try:
        config = get_config()
        if "metadata" in config:
            sandbox_id = config["metadata"].get("sandbox_id")
            if isinstance(sandbox_id, str) and sandbox_id:
                return sandbox_id
    except Exception:
        pass
    return None


async def get_sandbox_backend(thread_id: str) -> Any | None:
    """Get sandbox backend from cache, or connect using thread metadata."""
    sandbox_backend = SANDBOX_BACKENDS.get(thread_id)
    if sandbox_backend:
        return sandbox_backend

    sandbox_id = await get_sandbox_id_from_metadata(thread_id)
    if not sandbox_id:
        raise ValueError(f"Missing sandbox_id in thread metadata for {thread_id}")

    sandbox_backend = await asyncio.to_thread(create_sandbox, sandbox_id)
    SANDBOX_BACKENDS[thread_id] = sandbox_backend
    return sandbox_backend


def get_sandbox_backend_sync(thread_id: str) -> Any | None:
    """Sync wrapper for get_sandbox_backend."""
    return asyncio.run(get_sandbox_backend(thread_id))
