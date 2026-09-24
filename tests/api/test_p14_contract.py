from fastapi.testclient import TestClient
import time

import agy_bridge.api.app as app_module
from agy_bridge.api.app import create_app
from agy_bridge.api.envelope import parse_envelope
from agy_bridge.engine.agent import render_prompt


class Engine:
    def __init__(self):
        self.calls = []

    def health(self):
        return {"ok": True}

    def execute_completion(self, request):
        self.calls.append(request)
        return {"content": "ok", "usage": {}}


EXPECTED_MODELS = [
    "gemini-3.8-flash-high", "gemini-3.8-flash-medium", "gemini-3.8-flash-low",
    "gemini-3.7-flash-high", "gemini-3.7-flash-medium", "gemini-3.7-flash-low",
    "gemini-3.6-flash-high", "gemini-3.6-flash-medium", "gemini-3.6-flash-low",
    "gemini-3.1-pro-high", "gemini-3.1-pro-low", "claude-sonnet-4-6",
    "claude-opus-4-6-thinking", "gpt-oss-120b-medium", "gemini-3.5-flash-lite",
    "gemini-3.5-flash", "gemini-flash", "gemini-pro", "agy", "agy-auto",
]


def test_p14_model_catalog_and_health_contract(monkeypatch):
    monkeypatch.setenv("AGY_FLASH_MODEL", "gemini-3.8-flash")
    engine = Engine()
    client = TestClient(create_app(engine))
    models = client.get("/v1/models").json()["data"]
    assert [item["id"] for item in models] == EXPECTED_MODELS
    assert len(models) == 20
    assert all(item["owned_by"] == "agy" for item in models)
    health = client.get("/health").json()
    assert health == {
        "status": "healthy", "app_lane": {"ok": True},
        "active_lanes": 0, "port": 8790,
    }


def test_p14_model_default_and_aliases(monkeypatch):
    monkeypatch.setenv("AGY_BRIDGE_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("AGY_FLASH_MODEL", "gemini-3.8-flash")
    engine = Engine()
    client = TestClient(create_app(engine))
    for requested, expected in ((None, "gemini-3.5-flash-lite"), ("flash", "gemini-3.8-flash-medium"),
                                ("pro", "gemini-3.8-flash-high"), ("agy", "gemini-3.8-flash"),
                                ("auto", "gemini-3.8-flash"), ("default", "gemini-3.8-flash"),
                                ("", "gemini-3.8-flash")):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        if requested is not None:
            body["model"] = requested
        client.post("/v1/chat/completions", json=body)
        assert engine.calls[-1]["model"] == expected


def test_p14_conversation_lifecycle_and_usage_shape():
    engine = Engine()
    client = TestClient(create_app(engine))
    created = client.post("/v1/conversations", json={"conversation_id": "lane-a"}).json()
    assert created == {"conversation_id": "lane-a", "status": "open", "model": "gemini-3.8-flash",
                       "active": 1, "pool_size": "app-lane (elastic)"}
    assert client.get("/v1/conversations").json()["active"] == 1
    response = client.post("/v1/chat/completions", json={"model": "gemini-3.8-flash-high",
        "messages": [{"role": "user", "content": "hi"}]}).json()
    assert response["usage"] == {}
    assert client.delete("/v1/conversations/lane-a").json() == {
        "conversation_id": "lane-a", "status": "closed", "active": 0,
    }


def test_p14_tool_envelope_response_shape():
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    content, calls = parse_envelope('{"content":null,"tool_calls":[{"name":"lookup","arguments":{"q":"x"}}]}', tools)
    assert content == ""
    assert calls == [{"name": "lookup", "arguments": {"q": "x"}}]


def test_p14_image_legacy_placeholders_and_tool_api_shape():
    prompt = render_prompt([{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
    ]}])
    assert "[Embedded Image data (25 bytes)]" in prompt
    assert "[Image URL: https://example.test/image.png]" in prompt

    class ToolEngine(Engine):
        def execute_completion(self, request):
            self.calls.append(request)
            return {"content": "", "tool_calls": [{"id": "call_p14", "type": "function",
                "function": {"name": "lookup", "arguments": '{"q":"x"}'}}], "finish_reason": "tool_calls", "usage": {}}

    client = TestClient(create_app(ToolEngine()))
    body = {"model": "gemini-3.8-flash-high", "messages": [{"role": "user", "content": "look up x"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}], "stream": True}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert '"tool_calls"' in response.text and '"finish_reason": "tool_calls"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_p14_stream_heartbeats_while_sync_engine_runs(monkeypatch):
    class SlowEngine(Engine):
        def execute_completion(self, request):
            time.sleep(0.08)
            return {"content": "after delay", "usage": {}}

    original_wait = app_module.asyncio.wait
    observed_timeout = []

    async def fast_test_wait(tasks, timeout):
        observed_timeout.append(timeout)
        return await original_wait(tasks, timeout=0.01)

    monkeypatch.setattr(app_module.asyncio, "wait", fast_test_wait)
    client = TestClient(create_app(SlowEngine()))
    response = client.post("/v1/chat/completions", json={"model": "gemini-3.8-flash-high",
        "messages": [{"role": "user", "content": "hi"}], "stream": True})
    frames = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert response.status_code == 200
    assert observed_timeout and all(timeout == 15.0 for timeout in observed_timeout)
    assert sum('"content": ""' in frame for frame in frames) >= 2
    assert frames[-1] == "data: [DONE]"
