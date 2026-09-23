"""Hermetic tests for the AGY gRPC-web+json transport (engine/transport.py).

The HTTP tests run against a scripted ThreadingHTTPServer on an ephemeral
loopback port: real sockets, real http.client paths, zero production surface.
Never touches :8790 or the production Docker container.
"""
from __future__ import annotations

import http.server
import json
import threading
import time

import pytest

from agy_bridge.engine.registry import EngineEndpoint
from agy_bridge.engine.transport import (
    AgyHttpTransport,
    ScriptedTransport,
    SubscriptionResult,
    SERVICE,
)
from agy_bridge.protocol.frames import encode_frame
from agy_bridge.errors import (
    RateLimitExceeded,
    UpstreamAuthRequired,
    UpstreamTimeout,
    UpstreamUnavailable,
    MalformedFrameError,
)


def make_endpoint(port: int, host: str = "127.0.0.1", csrf: str = "tok") -> EngineEndpoint:
    return EngineEndpoint(
        account_id="test",
        host=host,
        http_port=port,
        csrf_secret=csrf,
        capability_profile="gemini-3.8-flash",
    )


def trailer_bytes(status: int = 0, message: str | None = None) -> bytes:
    lines = [f"grpc-status: {status}"]
    if message is not None:
        lines.append(f"grpc-message: {message}")
    return encode_frame(("\r\n".join(lines) + "\r\n").encode("utf-8"), trailer=True)


class ScriptedGrpcHandler(http.server.BaseHTTPRequestHandler):
    """Serves scripted per-path byte chunks like the language server."""

    script: dict = {}
    requests: list = []

    def log_message(self, *args) -> None:  # silence
        pass

    def _serve(self) -> None:
        entry = type(self).script.get(self.path)
        if entry is None:
            self.send_response(404)
            self.end_headers()
            return
        body = self.rfile.read(int(self.headers.get("content-length", "0") or 0))
        type(self).requests.append(
            {
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": body,
            }
        )
        self.send_response(int(entry.get("status", 200)))
        if entry.get("grpc_status") is not None:
            self.send_header("grpc-status", str(entry["grpc_status"]))
        if entry.get("grpc_message"):
            self.send_header("grpc-message", entry["grpc_message"])
        self.send_header("content-type", "application/grpc-web+json")
        self.end_headers()
        for chunk in entry.get("responses", []):
            self.wfile.write(chunk)
            self.wfile.flush()
            time.sleep(float(entry.get("delay", 0)))

    do_POST = _serve
    do_GET = _serve


@pytest.fixture
def grpc_server():
    handler = type("Handler", (ScriptedGrpcHandler,), {"script": {}, "requests": []})
    # Bind all interfaces so the Host-spoof test can connect via 127.0.0.2.
    server = http.server.ThreadingHTTPServer(("", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, handler
    server.shutdown()
    server.server_close()


def data_frame(payload: dict) -> bytes:
    return encode_frame(json.dumps(payload).encode("utf-8"))


# ---------------------------------------------------------------------------
# unary
# ---------------------------------------------------------------------------


def test_unary_happy_path(grpc_server):
    server, handler = grpc_server
    handler.script["/exa.language_server_pb.LanguageServerService/StartCascade"] = {
        "responses": [data_frame({"cascadeId": "c-123"}), trailer_bytes()]
    }
    transport = AgyHttpTransport()
    ep = make_endpoint(server.server_address[1])
    result = transport.unary(ep, "StartCascade", {"source": "CASCADE_CLIENT", "prompt": "hi"})
    assert result == {"cascadeId": "c-123"}


def test_unary_request_shape(grpc_server):
    server, handler = grpc_server
    method = "StartCascade"
    handler.script[f"/exa.language_server_pb.LanguageServerService/{method}"] = {
        "responses": [data_frame({"cascadeId": "c"}), trailer_bytes()]
    }
    transport = AgyHttpTransport()
    ep = make_endpoint(server.server_address[1], csrf="secret-csrf")
    transport.unary(ep, method, {"prompt": "hello"}, timeout=30)
    req = handler.requests[0]
    assert req["path"] == "/exa.language_server_pb.LanguageServerService/StartCascade"
    headers = req["headers"]
    assert headers["content-type"] == "application/grpc-web+json"
    assert headers["x-grpc-web"] == "1"
    assert headers["x-codeium-csrf-token"] == "secret-csrf"
    # body is a single gRPC-web data frame carrying the JSON payload
    assert req["body"][0] == 0x00
    length = int.from_bytes(req["body"][1:5], "big")
    payload = json.loads(req["body"][5 : 5 + length])
    assert payload == {"prompt": "hello"}


def test_unary_host_spoof_for_non_loopback(grpc_server):
    """Non-loopback hosts claim a loopback Host header (brain loopback gate)."""
    server, handler = grpc_server
    handler.script["/exa.language_server_pb.LanguageServerService/GetUserStatus"] = {
        "responses": [data_frame({"userStatus": {"email": "x@y.z"}}), trailer_bytes()]
    }
    transport = AgyHttpTransport()
    # Machine LAN IP: resolvable, non-loopback, reachable via the all-interfaces
    # bind; the transport must spoof a 127.0.0.1 Host header (the real
    # language server's loopback gate requires it for container endpoints).
    import socket as _socket

    probe = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        lan_ip = probe.getsockname()[0]
    finally:
        probe.close()
    ep = make_endpoint(server.server_address[1], host=lan_ip)
    transport.unary(ep, "GetUserStatus", {})
    assert handler.requests[0]["headers"]["host"] == f"127.0.0.1:{server.server_address[1]}"


@pytest.mark.parametrize(
    "grpc_code,exc_type",
    [
        (4, UpstreamTimeout),          # DEADLINE_EXCEEDED
        (8, RateLimitExceeded),        # RESOURCE_EXHAUSTED
        (14, UpstreamUnavailable),     # UNAVAILABLE
        (16, UpstreamAuthRequired),    # UNAUTHENTICATED
    ],
)
def test_unary_grpc_status_mapping(grpc_server, grpc_code, exc_type):
    server, handler = grpc_server
    handler.script["/exa.language_server_pb.LanguageServerService/StartCascade"] = {
        "responses": [data_frame({}), trailer_bytes(grpc_code, "upstream said no")]
    }
    transport = AgyHttpTransport()
    ep = make_endpoint(server.server_address[1])
    with pytest.raises(exc_type):
        transport.unary(ep, "StartCascade", {})


def test_unary_http_error_maps_unavailable(grpc_server):
    server, handler = grpc_server
    handler.script["/exa.language_server_pb.LanguageServerService/StartCascade"] = {
        "status": 503,
        "grpc_message": "service gone",
    }
    transport = AgyHttpTransport()
    ep = make_endpoint(server.server_address[1])
    with pytest.raises(UpstreamUnavailable):
        transport.unary(ep, "StartCascade", {})


def test_unary_connection_refused_maps_unavailable():
    # Bind a socket, note the port, close it: nothing listens there.
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    transport = AgyHttpTransport(connect_timeout=2.0)
    ep = make_endpoint(port)
    with pytest.raises(UpstreamUnavailable):
        transport.unary(ep, "StartCascade", {})


def test_unary_no_data_frames_returns_empty(grpc_server):
    server, handler = grpc_server
    handler.script["/exa.language_server_pb.LanguageServerService/CancelCascadeInvocation"] = {
        "responses": [trailer_bytes()]
    }
    transport = AgyHttpTransport()
    ep = make_endpoint(server.server_address[1])
    assert transport.unary(ep, "CancelCascadeInvocation", {"cascadeId": "c"}) == {}


# ---------------------------------------------------------------------------
# subscribe
# ---------------------------------------------------------------------------


def test_subscribe_delivers_frames_and_terminal_outcome(grpc_server):
    server, handler = grpc_server
    handler.script["/exa.language_server_pb.LanguageServerService/StreamAgentStateUpdates"] = {
        "responses": [
            data_frame({"update": {"status": "CASCADE_RUN_STATUS_RUNNING"}}),
            data_frame({"update": {"fullyIdle": True}}),
            trailer_bytes(),
        ]
    }
    transport = AgyHttpTransport()
    ep = make_endpoint(server.server_address[1])
    received = []
    result = transport.subscribe(
        ep,
        "StreamAgentStateUpdates",
        {"conversationId": "c1", "subscriberId": "s1"},
        on_frame=received.append,
        timeout=30,
    )
    assert len(received) == 2
    assert received[0]["update"]["status"] == "CASCADE_RUN_STATUS_RUNNING"
    assert result.outcome == SubscriptionResult.COMPLETED
    assert result.trailers is not None and result.trailers.is_success
    assert result.error is None


def test_subscribe_grpc_timeout_header(grpc_server):
    server, handler = grpc_server
    handler.script["/exa.language_server_pb.LanguageServerService/StreamAgentStateUpdates"] = {
        "responses": [trailer_bytes()]
    }
    transport = AgyHttpTransport()
    ep = make_endpoint(server.server_address[1])
    transport.subscribe(
        ep, "StreamAgentStateUpdates", {"conversationId": "c"}, on_frame=lambda d: None, timeout=90
    )
    assert handler.requests[0]["headers"].get("grpc-timeout") == "90000m"


def test_subscribe_eof_without_trailer_is_incomplete(grpc_server):
    server, handler = grpc_server
    handler.script["/exa.language_server_pb.LanguageServerService/StreamAgentStateUpdates"] = {
        "responses": [data_frame({"update": {"status": "CASCADE_RUN_STATUS_RUNNING"}})]
    }
    transport = AgyHttpTransport()
    ep = make_endpoint(server.server_address[1])
    received = []
    result = transport.subscribe(
        ep,
        "StreamAgentStateUpdates",
        {"conversationId": "c1", "subscriberId": "s1"},
        on_frame=received.append,
        timeout=30,
    )
    assert len(received) == 1
    assert result.outcome == SubscriptionResult.TERMINATED
    assert result.error is not None


def test_subscribe_on_frame_stop_halts_collection(grpc_server):
    server, handler = grpc_server
    handler.script["/exa.language_server_pb.LanguageServerService/StreamAgentStateUpdates"] = {
        "responses": [
            data_frame({"n": 1}),
            data_frame({"n": 2}),
            trailer_bytes(),
        ]
    }
    transport = AgyHttpTransport()
    ep = make_endpoint(server.server_address[1])
    received = []

    def stop_after_first(data):
        received.append(data)
        return True

    result = transport.subscribe(
        ep, "StreamAgentStateUpdates", {}, on_frame=stop_after_first, timeout=30
    )
    assert received == [{"n": 1}]
    assert result.outcome == SubscriptionResult.CANCELLED


def test_subscribe_grpc_status_header_fails_immediately(grpc_server):
    server, handler = grpc_server
    handler.script["/exa.language_server_pb.LanguageServerService/StreamAgentStateUpdates"] = {
        "grpc_status": 8,
        "grpc_message": "Resource exhausted",
    }
    transport = AgyHttpTransport()
    ep = make_endpoint(server.server_address[1])
    with pytest.raises(RateLimitExceeded):
        transport.subscribe(ep, "StreamAgentStateUpdates", {}, on_frame=lambda d: None, timeout=30)


def test_subscribe_malformed_frame_flags_raise(grpc_server):
    server, handler = grpc_server
    # 0x01 is a reserved flag: must be rejected with MalformedFrameError
    handler.script["/exa.language_server_pb.LanguageServerService/StreamAgentStateUpdates"] = {
        "responses": [b"\x01\x00\x00\x00\x02{}"]
    }
    transport = AgyHttpTransport()
    ep = make_endpoint(server.server_address[1])
    with pytest.raises(MalformedFrameError):
        transport.subscribe(ep, "StreamAgentStateUpdates", {}, on_frame=lambda d: None, timeout=30)


# ---------------------------------------------------------------------------
# ScriptedTransport
# ---------------------------------------------------------------------------


def test_scripted_unary_returns_payload_and_records_calls():
    t = ScriptedTransport(script={"StartCascade": {"cascadeId": "c-scripted"}})
    result = t.unary(make_endpoint(1), "StartCascade", {"prompt": "hi"})
    assert result == {"cascadeId": "c-scripted"}
    assert t.calls == [("StartCascade", {"prompt": "hi"})]


def test_scripted_unary_default_empty():
    t = ScriptedTransport()
    assert t.unary(make_endpoint(1), "Unknown", {}) == {}


def test_scripted_unary_raises_scripted_error():
    t = ScriptedTransport(script={"StartCascade": UpstreamUnavailable("nope")})
    with pytest.raises(UpstreamUnavailable):
        t.unary(make_endpoint(1), "StartCascade", {})


def test_scripted_subscribe_runs_frames_and_trailer():
    t = ScriptedTransport(
        script={
            "StreamAgentStateUpdates": [
                {"frame": {"a": 1}},
                {"frame": {"a": 2}},
                {"trailer": {"status": 0}},
            ]
        }
    )
    received = []
    result = t.subscribe(
        make_endpoint(1), "StreamAgentStateUpdates", {}, on_frame=received.append, timeout=30
    )
    assert received == [{"a": 1}, {"a": 2}]
    assert result.outcome == SubscriptionResult.COMPLETED


def test_scripted_subscribe_without_trailer_is_terminated():
    t = ScriptedTransport(script={"StreamAgentStateUpdates": [{"frame": {"a": 1}}]})
    received = []
    result = t.subscribe(
        make_endpoint(1), "StreamAgentStateUpdates", {}, on_frame=received.append, timeout=30
    )
    assert result.outcome == SubscriptionResult.TERMINATED


def test_scripted_subscribe_raises_scripted_error():
    t = ScriptedTransport(
        script={"StreamAgentStateUpdates": [{"error": RateLimitExceeded("quota out")}]}
    )
    with pytest.raises(RateLimitExceeded):
        t.subscribe(make_endpoint(1), "StreamAgentStateUpdates", {}, on_frame=lambda d: None, timeout=30)