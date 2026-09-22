"""Characterization tests for bridge.GatewayHandler HTTP surface.

Spin an in-process ThreadingHTTPServer on 127.0.0.1:0 (random free port) and
monkeypatch bridge._app_lane_turn + bridge._get_app_manager so no network is
ever touched. These tests freeze the CURRENT wire bytes / JSON shapes; they
must pass against the current handler before any refactor.
"""
import json
import threading
from http.server import ThreadingHTTPServer
from urllib import request as urlrequest
from urllib.error import HTTPError

import pytest

import bridge
from tests.conftest import FakeAppLaneManager


@pytest.fixture
def server(monkeypatch):
    """Start a GatewayHandler server on a random port with fake app lane."""
    real_turn = bridge._app_lane_turn
    mgr = FakeAppLaneManager()

    def fake_turn(conversation_id, user_text, timeout=180.0, model_enum=None, on_heartbeat=None):
        # Test sets mgr._response; generic fake returns recorded text.
        text, usage = mgr._response
        mgr.calls.append(("_app_lane_turn", conversation_id, user_text, model_enum))
        return text, dict(usage)

    monkeypatch.setattr(bridge, "_app_lane_turn", fake_turn)
    monkeypatch.setattr(bridge, "_get_app_manager", lambda: mgr)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), bridge.GatewayHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield {"base": f"http://127.0.0.1:{port}", "mgr": mgr, "httpd": httpd}
    httpd.shutdown()
    httpd.server_close()


def _get(server, path):
    try:
        with urlrequest.urlopen(f'{server["base"]}{path}', timeout=10) as r:
            return r.status, r.read().decode("utf-8"), r.headers
    except HTTPError as e:
        return e.code, e.read().decode("utf-8"), e.headers


def _post(server, path, body, headers=None):
    data = json.dumps(body).encode("utf-8")
    req = urlrequest.Request(f'{server["base"]}{path}', data=data,
                             headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urlrequest.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8"), r.headers
    except HTTPError as e:
        return e.code, e.read().decode("utf-8"), e.headers


def _delete(server, path):
    req = urlrequest.Request(f'{server["base"]}{path}', method="DELETE")
    try:
        with urlrequest.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8"), r.headers
    except HTTPError as e:
        return e.code, e.read().decode("utf-8"), e.headers


# ---------------------------------------------------------------------------
# GET endpoints
# ---------------------------------------------------------------------------

def test_get_v1_models(server):
    status, body, hdrs = _get(server, "/v1/models")
    assert status == 200
    data = json.loads(body)
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert ids == bridge.MODELS
    assert all(m["object"] == "model" and m["owned_by"] == "agy" for m in data["data"])


def test_get_health_healthy_shape(server):
    server["mgr"]._health = {"ok": True, "email": "fake@example.com", "port": 9,
                             "source": "registry", "lane": "main", "lanes": []}
    status, body, _ = _get(server, "/health")
    assert status == 200
    data = json.loads(body)
    assert data["status"] == "healthy"
    assert data["app_lane"]["ok"] is True
    assert data["port"] == bridge.PORT


def test_get_unknown_route_404(server):
    status, body, _ = _get(server, "/nope")
    assert status == 404
    data = json.loads(body)
    assert data["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# DELETE endpoints
# ---------------------------------------------------------------------------

def test_delete_conversation_closes_lane(server):
    server["mgr"].acquire_for_lane("abc123")
    status, body, _ = _delete(server, "/v1/conversations/abc123")
    assert status == 200
    data = json.loads(body)
    assert data["conversation_id"] == "abc123"
    assert data["status"] == "closed"
    assert "abc123" not in server["mgr"].lanes()


def test_delete_conversation_missing_id_400(server):
    status, body, _ = _delete(server, "/v1/conversations/")
    assert status == 400
    assert json.loads(body)["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — non-streaming
# ---------------------------------------------------------------------------

def test_post_chat_non_streaming_content(server):
    server["mgr"]._response = ("Hello from fake lane", {"prompt_tokens": 5})
    status, body, _ = _post(server, "/v1/chat/completions",
                            {"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200
    data = json.loads(body)
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello from fake lane"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {"prompt_tokens": 5}


def test_post_chat_tool_call_passthrough(server):
    # Non-streaming tool call: fake lane returns envelope text with a tool call.
    server["mgr"]._response = ('{"content":"", "tool_calls":[{"name":"lookup_weather","arguments":{"city":"X"}}]}', {})
    status, body, _ = _post(
        server, "/v1/chat/completions",
        {"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "weather"}],
         "tools": [{"type": "function", "function": {"name": "lookup_weather"}}],
         "tool_choice": "required"})
    assert status == 200
    data = json.loads(body)
    msg = data["choices"][0]["message"]
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["function"]["name"] == "lookup_weather"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"city": "X"}


def test_post_chat_conversation_lane(server):
    server["mgr"]._response = ("lane reply", {})
    status, body, _ = _post(
        server, "/v1/chat/completions",
        {"model": "gemini-3.8-flash", "conversation_id": "conv-1",
         "messages": [{"role": "user", "content": "lateral text"}]})
    assert status == 200
    data = json.loads(body)
    assert data["choices"][0]["message"]["content"] == "lane reply"
    # _app_lane_turn is monkeypatched, so the real chat_lane() lane registration
    # is bypassed; assert the conversation_id was routed to the fake turn.
    assert any(c[0] == "_app_lane_turn" and c[1] == "conv-1" for c in server["mgr"].calls)


def test_post_chat_conversation_lane_requires_user_message(server):
    status, body, _ = _post(
        server, "/v1/chat/completions",
        {"model": "gemini-3.8-flash", "conversation_id": "conv-x",
         "messages": [{"role": "system", "content": "no user msg"}]})
    assert status == 400
    assert json.loads(body)["error"]["type"] == "invalid_request_error"


def test_post_chat_error_maps_to_502(server, monkeypatch):
    def boom(conversation_id, user_text, timeout=180.0, model_enum=None, on_heartbeat=None):
        raise RuntimeError("upstream exploded")
    monkeypatch.setattr(bridge, "_app_lane_turn", boom)
    status, body, _ = _post(server, "/v1/chat/completions",
                            {"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 502
    data = json.loads(body)
    assert data["error"]["type"] == "upstream_error"
    assert "upstream exploded" in data["error"]["message"]


def test_post_bad_json_400(server):
    req = urlrequest.Request(f'{server["base"]}/v1/chat/completions',
                             data=b"{not json", headers={"Content-Type": "application/json"})
    try:
        urlrequest.urlopen(req, timeout=10)
        pytest.fail("expected 400")
    except HTTPError as e:
        assert e.code == 400
        assert json.loads(e.read())["error"]["type"] == "invalid_request_error"


def test_post_unknown_route_404(server):
    status, body, _ = _post(server, "/v1/whatever", {"a": 1})
    assert status == 404
    assert json.loads(body)["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# POST streaming (SSE)
# ---------------------------------------------------------------------------

def _parse_sse(body):
    events = []
    for line in body.split("\n"):
        if line.startswith("data: "):
            events.append(line[6:])
    return events


def test_post_chat_streaming_sse_done_terminator(server):
    server["mgr"]._response = ("Short reply", {})
    status, body, _ = _post(server, "/v1/chat/completions",
                            {"model": "gemini-3.8-flash", "stream": True,
                             "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200
    events = _parse_sse(body)
    assert events[-1] == "[DONE]"
    # content chunks appear in order
    chunks = []
    for ev in events[:-1]:
        d = json.loads(ev)
        c = d["choices"][0].get("delta", {}).get("content")
        if c:
            chunks.append(c)
    assert "".join(chunks) == "Short reply"


def test_post_chat_streaming_tool_calls(server):
    server["mgr"]._response = ('{"content":"", "tool_calls":[{"name":"tool_a","arguments":{"p":1}}]}', {})
    status, body, _ = _post(
        server, "/v1/chat/completions",
        {"model": "gemini-3.8-flash", "stream": True,
         "messages": [{"role": "user", "content": "go"}],
         "tools": [{"type": "function", "function": {"name": "tool_a"}}]})
    assert status == 200
    events = _parse_sse(body)
    assert events[-1] == "[DONE]"
    tool_event = None
    for ev in events:
        d = json.loads(ev)
        delta = d["choices"][0].get("delta", {})
        if delta.get("tool_calls"):
            tool_event = delta["tool_calls"][0]
            break
    assert tool_event is not None
    assert tool_event["function"]["name"] == "tool_a"
    assert json.loads(tool_event["function"]["arguments"]) == {"p": 1}


def test_post_chat_streaming_error_inline(server, monkeypatch):
    def boom(conversation_id, user_text, timeout=180.0, model_enum=None, on_heartbeat=None):
        raise RuntimeError("stream fail")
    monkeypatch.setattr(bridge, "_app_lane_turn", boom)
    status, body, _ = _post(server, "/v1/chat/completions",
                            {"model": "gemini-3.8-flash", "stream": True,
                             "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200
    events = _parse_sse(body)
    assert events[-1] == "[DONE]"
    assert any("stream fail" in ev for ev in events)