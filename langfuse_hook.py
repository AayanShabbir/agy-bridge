#!/usr/bin/env python3
"""langfuse_hook.py — FAIL-OPEN LangFuse ingestion emitter for the AGY bridge.

Sits at bridge.py:_app_lane_turn() (the single choke point every /v1/chat/completions
call passes through) and posts one generation trace to LangFuse per AI run.

CONTRACT (this hook):
- Emits a LangFuse ingestion v2 batch: one trace-create + one generation-create.
- Captures model, input, output, latency, status (success/error), error_class,
  error message, conversation_id, and a fresh trace_id.
- FAIL-OPEN, NON-BLOCKING: any exception or network failure here is swallowed and
  logged to stderr. The inference path is NEVER blocked or slowed by telemetry.
- stdlib-only (urllib + hmac + json) so it matches the bridge container's
  zero-extra-deps build. No image rebuild required (file is bind-mounted).

ENV / creds resolution (order):
  1. os.environ[LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL]
  2. a mounted .env file at LANGFUSE_ENV_FILE (default /root/.hermes/.env in the
     container), parsed with a minimal stdlib loader. (The compose already mounts
     /Users/aayan/.hermes/.env -> /root/.hermes/.env:ro.)
  The BASE URL inside the container must be reachable from it:
  default http://host.docker.internal:3000 (LangFuse UI publishes 0.0.0.0:3000 on the host).

Ingestion auth (LangFuse 4.38, verified against official python SDK 3.7.0):
  POST {base}/api/public/ingestion
  headers: Authorization: Basic base64("<public_key>:<secret_key>"),
           x_langfuse_sdk_name, x_langfuse_sdk_version, x_langfuse_public_key,
           Content-Type: application/json
  (Basic auth replaces the pre-v3 X-Langfuse-Signature HMAC scheme; HMAC headers
   return 401 on modern LangFuse.)
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
import urllib.request
import urllib.error
from base64 import b64encode
from datetime import datetime, timezone

SDK_NAME = "python"  # kept identical to the official SDK so server accepts it
SDK_VERSION = "3.7.0"
DEFAULT_ENV_FILE = os.environ.get("LANGFUSE_ENV_FILE", "/root/.hermes/.env")
DEFAULT_BASE_URL = "http://host.docker.internal:3000"


def _load_env_file(path: str) -> dict:
    """Minimal stdlib KEY=VALUE .env parser (no dotenv dep). Skips comments/blank."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("\"'")
                if k:
                    out[k] = v
    except OSError:
        pass
    return out


def _resolve_config() -> dict:
    cfg = {
        "public_key": os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        "secret_key": os.environ.get("LANGFUSE_SECRET_KEY", ""),
        "base_url": os.environ.get("LANGFUSE_BASE_URL", "")
                    or os.environ.get("LANGFUSE_BASEURL", "")
                    or DEFAULT_BASE_URL,
        "enabled": os.environ.get("LANGFUSE_ENABLED", "1") not in ("0", "false", "False", ""),
    }
    if not (cfg["public_key"] and cfg["secret_key"]):
        env = _load_env_file(DEFAULT_ENV_FILE)
        cfg["public_key"] = cfg["public_key"] or env.get("LANGFUSE_PUBLIC_KEY", "")
        cfg["secret_key"] = cfg["secret_key"] or env.get("LANGFUSE_SECRET_KEY", "")
        cfg["base_url"] = env.get("LANGFUSE_BASE_URL") or cfg["base_url"]
    return cfg


def _iso_z(ts: float | None = None) -> str:
    t = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _emit(trace: dict, generation: dict) -> None:
    """POST a trace-create + generation-create batch to LangFuse ingestion. Raises on failure."""
    cfg = _resolve_config()
    if not cfg["enabled"] or not (cfg["public_key"] and cfg["secret_key"]):
        raise RuntimeError("langfuse disabled or keys missing (enabled=%s)" % cfg["enabled"])

    now_ms = int(time.time() * 1000)
    batch = {
        "batch": [
            {
                "id": str(uuid.uuid4()),
                "type": "trace-create",
                "timestamp": _iso_z(),
                "body": trace,
            },
            {
                "id": str(uuid.uuid4()),
                "type": "generation-create",
                "timestamp": _iso_z(),
                "body": generation,
            },
        ]
    }
    body_bytes = json.dumps(batch, separators=(",", ":")).encode("utf-8")
    basic = b64encode(f"{cfg['public_key']}:{cfg['secret_key']}".encode("utf-8")).decode("ascii")

    url = cfg["base_url"].rstrip("/") + "/api/public/ingestion"
    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers={
            "Authorization": "Basic " + basic,
            "Content-Type": "application/json",
            "x_langfuse_sdk_name": SDK_NAME,
            "x_langfuse_sdk_version": SDK_VERSION,
            "x_langfuse_public_key": cfg["public_key"],
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        code = resp.status
        if code == 200:
            return
        if code == 207:
            # v4 ingestion returns 207 per-event results — succeed only if no
            # per-event errors (mirrors the official SDK's _process_response).
            errs = (json.loads(resp.read().decode() or "{}") or {}).get("errors") or []
            if not errs:
                return
            raise RuntimeError("langfuse ingestion 207 errors: %s" % errs[:2])
        raise RuntimeError("langfuse ingestion HTTP %s" % code)


_RING_DB_PATH = os.environ.get("JEV_RING_DB", "/state/jev_ring.db")
_RING_ENABLED = os.environ.get("JEV_RING_ENABLED", "1") != "0"

_ERR_BUCKET_MAP = {
    "timeout": {
        "socket.timeout", "TimeoutError", "asyncio.TimeoutError",
        "requests.exceptions.Timeout", "urllib3.exceptions.ReadTimeoutError",
        "httpx.ReadTimeout", "httpx.ConnectTimeout",
    },
    "quota_429": set(),  # HTTPError code==429 special-cased
    "connection_refused": {
        "ConnectionRefusedError", "ConnectionError", "ConnectionResetError",
        "BrokenPipeError", "http.client.RemoteDisconnected",
        "urllib3.exceptions.NewConnectionError", "httpx.ConnectError",
    },
    "no_planner_text": {"JSONDecodeError", "json.decoder.JSONDecodeError", "KeyError", "ValueError"},
}
_RAW_TO_BUCKET = {c: b for b, cs in _ERR_BUCKET_MAP.items() for c in cs}


def _bucket_error(raw_error_class, error_text=None, latency_ms=None) -> str:
    """Deterministic error bucketing. Empty string = healthy (no error)."""
    if raw_error_class is None or raw_error_class in ("", "None"):
        return "slow_response" if latency_ms is not None and latency_ms > 30000 else ""
    if raw_error_class in ("urllib.error.HTTPError", "HTTPError") and error_text and " 429" in f" {error_text}":
        return "quota_429"
    b = _RAW_TO_BUCKET.get(raw_error_class)
    if b:
        return b
    low = raw_error_class.lower()
    if "timeout" in low:
        return "timeout"
    if "connection" in low or "refused" in low or "disconnect" in low or "broken" in low:
        return "connection_refused"
    return "unknown"


def _ring_sqlite_ts(ts: float) -> str:
    """SQLite-compatible UTC timestamp (space-separated, seconds) so the
    state builder's datetime('now', ...) windows compare correctly lexically."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# SQLite connection is per-thread: the bridge is a ThreadingHTTPServer and a
# module-global "sqlite3.Connection" is only usable in the thread that created it
# ("SQLite objects created in a thread can only be used in that same thread").
# threading.local() gives each worker its own handle; DELETE journal + busy_timeout
# already make concurrent writers safe, so this is correctness without contention.
_ring_tls = threading.local()


def _ring_conn() -> sqlite3.Connection:
    """Get (and lazily initialize) this thread's ring-buffer connection."""
    conn = getattr(_ring_tls, "conn", None)
    if conn is not None:
        return conn
    conn = sqlite3.connect(_RING_DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS jev_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lane TEXT NOT NULL DEFAULT 'primary',
            port INTEGER NOT NULL DEFAULT 8790,
            daemon TEXT NOT NULL DEFAULT 'agy-bridge',
            model TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('success','error')),
            error_class TEXT NOT NULL DEFAULT '',
            timestamp_utc TEXT NOT NULL,
            trace_id TEXT NOT NULL DEFAULT '',
            gateway_decision TEXT NOT NULL DEFAULT ''
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jev_ts ON jev_events(timestamp_utc)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jev_status ON jev_events(status, timestamp_utc)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jev_err ON jev_events(error_class, timestamp_utc)")
    conn.commit()
    _ring_tls.conn = conn
    return conn


def _write_ring(*, model, latency_ms, status, error_class, error_text, timestamp, trace_id) -> None:
    """Append one inference event to the SQLite ring buffer (JEV's decision window).
    FAIL-OPEN: never raise; never block inference. Uses default (DELETE) journal mode
    for cross-UID robustness (container root writes, host aayan reads) — WAL's -shm
    shared-memory file is the classic root-vs-user trap. Dir/db must be chmod 666."""
    global _RING_DB_PATH, _RING_ENABLED
    if not _RING_ENABLED:
        return
    try:
        if not model or status not in ("success", "error"):
            return
        conn = _ring_conn()
        bucketed = _bucket_error(error_class, error_text=error_text, latency_ms=latency_ms)
        conn.execute(
            "INSERT INTO jev_events (model, latency_ms, status, error_class, timestamp_utc, trace_id) "
            "VALUES (?,?,?,?,?,?)",
            (model, latency_ms, status, bucketed, timestamp, trace_id),
        )
        conn.execute("DELETE FROM jev_events WHERE timestamp_utc < datetime('now','-1 hour')")
        conn.execute(
            "DELETE FROM jev_events WHERE id NOT IN "
            "(SELECT id FROM jev_events ORDER BY id DESC LIMIT 2000)"
        )
        conn.commit()
    except Exception as ex:
        print("[langfuse_hook] ring write FAILED (inference unaffected): %s" % ex, flush=True)


def emit_call(*,
              model: str,
              prompt: str,
              output: str,
              start_time: float,
              end_time: float,
              status: str = "success",
              error: str | None = None,
              error_class: str | None = None,
              conversation_id: str | None = None,
              usage: dict | None = None) -> str:
    """Emit one trace+generation for a finished AI call. Returns the trace_id.
    NEVER raises into the caller: every failure is caught and logged to stderr.
    """
    trace_id = str(uuid.uuid4())
    gen_id = str(uuid.uuid4())
    try:
        _write_ring(
            model=model,
            latency_ms=int(max(0.0, (end_time - start_time)) * 1000),
            status=status,
            error_class=error_class,
            error_text=error,
            timestamp=_ring_sqlite_ts(start_time),
            trace_id=trace_id,
        )
    except Exception as ex:
        # FAIL-OPEN: a ring-buffer failure must never touch the inference path.
        print("[langfuse_hook] ring write FAILED (inference unaffected): %s" % ex, flush=True)
    trace = {
        "id": trace_id,
        "timestamp": _iso_z(start_time),
        "name": "agy-bridge-call",
        "environment": "prod",
        "metadata": {
            "conversation_id": conversation_id,
            "source": "fleet",
        },
    }
    generation = {
        "id": gen_id,
        "traceId": trace_id,
        "name": "agy-bridge-call",
        "model": model,
        "input": prompt,
        "output": output,
        "usage": usage or {},
        "startTime": _iso_z(start_time),
        "endTime": _iso_z(end_time),
        "status": status,
        "metadata": {
            "conversation_id": conversation_id,
            "duration_s": round(max(0.0, end_time - start_time), 4),
        },
    }
    if status != "success":
        generation["metadata"]["error"] = error or ""
        generation["metadata"]["error_class"] = error_class or ""
    try:
        _emit(trace, generation)
    except Exception as ex:
        # FAIL-OPEN: telemetry must never break or slow the inference path.
        print("[langfuse_hook] emit FAILED (inference unaffected): %s" % ex, flush=True)
    return trace_id
