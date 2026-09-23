"""Layer 2: Real AGY gRPC-web+json transport over stdlib HTTP.

Implements the wire protocol verified against the live Antigravity language
server (see production app_lane.py): POST /exa.language_server_pb.LanguageServerService/<Method>
with content-type application/grpc-web+json, the x-grpc-web header flag, the
per-endpoint CSRF token, loopback Host spoofing for non-loopback endpoints,
and a grpc-timeout header on streaming calls. Response bodies are sequences
of gRPC-web frames (5-byte header + JSON payload) terminated by a trailer
frame carrying grpc-status.

The transport is a thin, typed adapter: no business decisions live here.
Failures are mapped to the typed error domain (UpstreamError family and
ProtocolError family) so engines can classify and react.

Hermetic: all behavior is testable against a scripted HTTP server; no
production surface (Docker :8790 or the Antigravity daemon) is ever touched
by this module or its tests.
"""
from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, cast

from agy_bridge.engine.registry import EngineEndpoint
from agy_bridge.errors import (
    MalformedFrameError,
    ProtocolError,
    RateLimitExceeded,
    UpstreamAuthRequired,
    UpstreamError,
    UpstreamTimeout,
    UpstreamUnavailable,
)
from agy_bridge.protocol.frames import Frame, FrameDecoder, encode_frame
from agy_bridge.protocol.messages import decode_json_frame
from agy_bridge.protocol.trailers import GrpcTrailers, parse_trailers

SERVICE = "exa.language_server_pb.LanguageServerService"
METHOD_PATH = f"/{SERVICE}/{{method}}"
CONTENT_TYPE = "application/grpc-web+json"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

# gRPC status codes mapped to the typed error domain
_GRPC_STATUS_ERRORS: Dict[int, Any] = {
    4: UpstreamTimeout,       # DEADLINE_EXCEEDED
    8: RateLimitExceeded,     # RESOURCE_EXHAUSTED
    14: UpstreamUnavailable,  # UNAVAILABLE
    16: UpstreamAuthRequired, # UNAUTHENTICATED
}
_GRPC_STATUS_DEFAULT = UpstreamError

# Transport-scoped operational errors (successful HTTP but no terminal evidence)
STREAM_COMPLETED = "completed"
STREAM_TERMINATED = "terminated"  # stream ended without a gRPC trailer
STREAM_FAILED = "failed"          # trailer with non-zero grpc-status
STREAM_CANCELLED = "cancelled"    # subscriber requested stop; not an error


class StreamSubscription(Protocol):
    """Opaque handle for an active subscription (test seam)."""

    def close(self) -> None: ...


@dataclass
class SubscriptionResult:
    """Outcome of a subscribe call, with the frames the subscriber consumed."""

    frames: List[Any] = field(default_factory=list)
    trailers: Optional[GrpcTrailers] = None
    outcome: str = STREAM_COMPLETED
    error: Optional[str] = None

    # Outcome constants (mirror the module-level STREAM_* values)
    COMPLETED = STREAM_COMPLETED
    TERMINATED = STREAM_TERMINATED
    FAILED = STREAM_FAILED
    CANCELLED = STREAM_CANCELLED


class EngineTransport(Protocol):
    """Minimal transport surface engines depend on (sync, stdlib-backed)."""

    def unary(self, endpoint: EngineEndpoint, method: str, payload: dict, *, timeout: float = 30.0) -> Any:
        """Issue a unary RPC, returning the decoded JSON data frame (or {})."""
        ...

    def subscribe(
        self,
        endpoint: EngineEndpoint,
        method: str,
        payload: dict,
        *,
        on_frame: Callable[[Any], Optional[bool]],
        timeout: float = 60.0,
    ) -> SubscriptionResult:
        """Stream data frames to on_frame; return the terminal outcome.

        on_frame may return True to stop early (cancelled, not an error).
        """
        ...


class AgyHttpTransport:
    """Production transport: real gRPC-web+json over http.client."""

    def __init__(
        self,
        *,
        host_override: Optional[str] = None,
        connect_timeout: float = 5.0,
        read_chunk_bytes: int = 65536,
    ) -> None:
        self.host_override = host_override
        self.connect_timeout = connect_timeout
        self.read_chunk_bytes = read_chunk_bytes

    # -- internals -----------------------------------------------------------

    def _endpoint(self, ep: EngineEndpoint) -> EngineEndpoint:
        if self.host_override:
            return EngineEndpoint(
                account_id=ep.account_id,
                host=self.host_override,
                http_port=ep.http_port,
                csrf_secret=ep.csrf_secret,
                engine_id=ep.engine_id,
                grpc_port=ep.grpc_port,
                generation=ep.generation,
                observed_at=ep.observed_at,
                capability_profile=ep.capability_profile,
            )
        return ep

    def _connect(self, ep: EngineEndpoint) -> http.client.HTTPConnection:
        host = self.host_override or ep.host
        try:
            return http.client.HTTPConnection(host, ep.http_port, timeout=self.connect_timeout)
        except (OSError, http.client.HTTPException) as ex:
            raise UpstreamUnavailable(f"Cannot reach language server at {host}:{ep.http_port}: {ex}") from ex

    @staticmethod
    def _headers(ep: EngineEndpoint) -> Dict[str, str]:
        headers = {
            "Content-Type": CONTENT_TYPE,
            "X-GRPC-Web": "1",
            "X-Codeium-CSRF-Token": ep.csrf_secret,
            "Accept": "*/*",
        }
        # The language server only accepts loopback origins; non-loopback
        # endpoints (containers) must claim a loopback Host header.
        if ep.host not in LOOPBACK_HOSTS:
            headers["Host"] = f"127.0.0.1:{ep.http_port}"
        return headers

    @staticmethod
    def _raise_for_trailers(trailers: Optional[GrpcTrailers], method: str) -> None:
        if trailers is None:
            return
        if not trailers.is_success:
            exc_cls = _GRPC_STATUS_ERRORS.get(trailers.status, _GRPC_STATUS_DEFAULT)
            base = trailers.message or f"gRPC status {trailers.status} from {method}"
            raise exc_cls(base)

    @staticmethod
    def _map_http_error(method: str, status: int, reason: str) -> UpstreamError:
        # Surface grpc-status headers when present (trailer-only responses)
        return UpstreamUnavailable(f"HTTP {status} {reason} from {method}")

    def _decode_http_status(self, method: str, status: int, reason: str, headers: Any) -> None:
        if 200 <= status < 300:
            # Trailer-only responses carry grpc-status in the HTTP headers.
            grpc_status = headers.get("grpc-status")
            if grpc_status is not None:
                trailers = parse_trailers(
                    f"grpc-status: {grpc_status}\r\ngrpc-message: {headers.get('grpc-message', '')}".encode()
                )
                self._raise_for_trailers(trailers, method)
            return
        grpc_status = headers.get("grpc-status")
        if grpc_status is not None:
            try:
                trailers = parse_trailers(
                    f"grpc-status: {grpc_status}\r\ngrpc-message: {headers.get('grpc-message', '')}".encode()
                )
            except Exception:
                trailers = None
            if trailers is not None:
                self._raise_for_trailers(trailers, method)
        raise self._map_http_error(method, status, reason)

    # -- RPC surface ---------------------------------------------------------

    def unary(self, endpoint: EngineEndpoint, method: str, payload: dict, *, timeout: float = 30.0) -> Any:
        ep = self._endpoint(endpoint)
        conn = self._connect(ep)
        try:
            body = encode_frame(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
            conn.request(
                "POST",
                METHOD_PATH.format(method=method),
                body=body,
                headers=self._headers(ep),
            )
            response = conn.getresponse()
            self._decode_http_status(method, response.status, response.reason, response.headers)
            raw = response.read()
        except (socket.timeout, TimeoutError) as ex:
            raise UpstreamTimeout(f"{method} timed out after {timeout}s") from ex
        except (OSError, http.client.HTTPException) as ex:
            raise UpstreamUnavailable(f"{method} failed at transport: {ex}") from ex
        finally:
            conn.close()

        decoder = FrameDecoder()
        frames = self._collect_frames(decoder, raw, method)
        data_frames = [f for f in frames if not f.is_trailer]
        trailers = None
        for f in frames:
            if f.is_trailer:
                trailers = parse_trailers(f.payload)
        self._raise_for_trailers(trailers, method)
        if not data_frames:
            return {}
        return decode_json_frame(data_frames[0])

    def subscribe(
        self,
        endpoint: EngineEndpoint,
        method: str,
        payload: dict,
        *,
        on_frame: Callable[[Any], Optional[bool]],
        timeout: float = 60.0,
    ) -> SubscriptionResult:
        ep = self._endpoint(endpoint)
        conn = self._connect(ep)
        decoder = FrameDecoder()
        result = SubscriptionResult()
        grpc_timeout_ms = max(1, int(timeout * 1000))
        try:
            body = encode_frame(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
            headers = self._headers(ep)
            headers["Grpc-Timeout"] = f"{grpc_timeout_ms}m"
            conn.request(
                "POST",
                METHOD_PATH.format(method=method),
                body=body,
                headers=headers,
            )
            response = conn.getresponse()
            self._decode_http_status(method, response.status, response.reason, response.headers)

            saw_trailer = False
            while True:
                chunk = response.read(self.read_chunk_bytes)
                if chunk:
                    for frame in decoder.feed(chunk):
                        if frame.is_trailer:
                            saw_trailer = True
                            trailers = parse_trailers(frame.payload)
                            result.trailers = trailers
                            if not trailers.is_success:
                                result.outcome = STREAM_FAILED
                                result.error = trailers.message or f"gRPC status {trailers.status}"
                                self._raise_for_trailers(trailers, method)
                            else:
                                result.outcome = STREAM_COMPLETED
                        else:
                            message = decode_json_frame(frame)
                            result.frames.append(message)
                            should_stop = on_frame(message)
                            if should_stop:
                                result.outcome = STREAM_CANCELLED
                                # Best-effort close of the upstream stream.
                                try:
                                    conn.close()
                                except Exception:
                                    pass
                                return result
                    continue

                # EOF reached
                if not saw_trailer:
                    # FrameDecoder validates trailing bytes
                    if decoder.finish() is not None:  # pragma: no cover - API stability
                        pass
                    result.outcome = STREAM_TERMINATED
                    result.error = (
                        f"{method} stream ended without a gRPC trailer "
                        "(no terminal evidence from upstream)"
                    )
                return result
        except (socket.timeout, TimeoutError) as ex:
            raise UpstreamTimeout(f"{method} stream timed out after {timeout}s") from ex
        except (OSError, http.client.HTTPException) as ex:
            raise UpstreamUnavailable(f"{method} stream failed at transport: {ex}") from ex
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _collect_frames(self, decoder: FrameDecoder, raw: bytes, method: str) -> List[Frame]:
        frames: List[Frame] = []
        if raw:
            frames.extend(decoder.feed(raw))
        if not frames:
            return frames
        # Validate the tail of the response is frame-aligned
        if len(decoder._buffer) > 0:  # pragma: no cover - decoder.finish covers
            decoder.finish()
        return frames


class ScriptedTransport:
    """Deterministic transport for hermetic engine tests.

    Script entries per method:
      - dict            -> returned wholesale by unary()
      - Exception       -> raised by both unary() and subscribe()
      - callable        -> f(payload) for unary, or f(payload, on_frame) for subscribe
      - list            -> subscribe script: items are {"frame": ...},
                           {"trailer": {"status": int}}, or {"error": Exception}
    """

    def __init__(self, script: Optional[Dict[str, Any]] = None, default: Any = None) -> None:
        self.script: Dict[str, Any] = script or {}
        self.default = default if default is not None else {}
        self.calls: List[tuple] = []

    def unary(self, endpoint: EngineEndpoint, method: str, payload: dict, *, timeout: float = 30.0) -> Any:
        self.calls.append((method, payload))
        entry = self.script.get(method, self.default)
        if isinstance(entry, Exception):
            raise entry
        if callable(entry):
            return entry(payload)
        return entry if isinstance(entry, dict) else {}

    def subscribe(
        self,
        endpoint: EngineEndpoint,
        method: str,
        payload: dict,
        *,
        on_frame: Callable[[Any], Optional[bool]],
        timeout: float = 60.0,
    ) -> SubscriptionResult:
        self.calls.append((method, payload))
        entry = self.script.get(method, self.default)
        if isinstance(entry, Exception):
            raise entry
        if callable(entry):
            return cast(SubscriptionResult, entry(payload, on_frame))
        result = SubscriptionResult(outcome=STREAM_TERMINATED, error="scripted transport: no trailer received")
        if not isinstance(entry, list):
            result.error = "scripted transport: no frames configured"
            return result
        for item in entry:
            if not isinstance(item, dict):
                continue
            if "error" in item:
                raise item["error"]
            if "trailer" in item:
                status = int(item["trailer"].get("status", 0))
                message = item["trailer"].get("message")
                result.trailers = GrpcTrailers(
                    status=status, message=message, is_success=(status == 0)
                )
                if status == 0:
                    result.outcome = STREAM_COMPLETED
                    result.error = None
                else:
                    result.outcome = STREAM_FAILED
                    result.error = message or f"gRPC status {status}"
                    exc_cls = _GRPC_STATUS_ERRORS.get(status, _GRPC_STATUS_DEFAULT)
                    raise exc_cls(result.error)
                continue
            frame = item.get("frame", item)
            result.frames.append(frame)
            if on_frame(frame):
                result.outcome = STREAM_CANCELLED
                return result
        return result


def error_kind(exc: BaseException) -> str:
    """Classify an upstream exception for scheduler cooldown bookkeeping."""
    if isinstance(exc, RateLimitExceeded):
        return "quota"
    if isinstance(exc, UpstreamAuthRequired):
        return "auth_required"
    if isinstance(exc, UpstreamTimeout):
        return "timeout"
    if isinstance(exc, UpstreamUnavailable):
        return "unavailable"
    if isinstance(exc, ProtocolError):
        return "protocol"
    return "upstream"