"""Tests for Phase 8: FastAPI Façade & Buffered SSE (Section 12)."""
import json
import pytest
from starlette.testclient import TestClient

from agy_bridge.api.app import create_app
from agy_bridge.errors import RateLimitExceeded, UpstreamUnavailable


class FakeCompletionEngine:
    def __init__(self, response_text: str = "Hello from Antigravity!", fails_with: Exception = None):
        self.response_text = response_text
        self.fails_with = fails_with
        self.requests = []

    def execute_completion(self, request_data: dict):
        self.requests.append(request_data)
        if self.fails_with:
            raise self.fails_with
        return {
            "content": self.response_text,
            "finish_reason": "stop",
            "model": request_data.get("model", "gemini-3.8-flash"),
        }


def test_health_endpoints():
    app = create_app(engine=FakeCompletionEngine())
    client = TestClient(app)

    r_live = client.get("/health/live")
    assert r_live.status_code == 200
    assert r_live.json() == {"status": "live"}

    r_ready = client.get("/health/ready")
    assert r_ready.status_code == 200
    assert r_ready.json() == {"status": "ready"}

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"


def test_models_endpoint():
    app = create_app(engine=FakeCompletionEngine())
    client = TestClient(app)

    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert "data" in data
    model_ids = [m["id"] for m in data["data"]]
    assert "gemini-3.8-flash" in model_ids
    assert "gemini-3.8-flash-high" in model_ids


def test_capabilities_endpoint():
    app = create_app(engine=FakeCompletionEngine())
    client = TestClient(app)

    r = client.get("/v1/bridge/capabilities")
    assert r.status_code == 200
    data = r.json()
    assert data["vision"]["policy"] == "reject"
    assert data["streaming"]["mode"] == "buffered_sse"
    assert data["conversations"]["supported"] is True


def test_conversations_crud():
    app = create_app(engine=FakeCompletionEngine())
    client = TestClient(app)

    # Create
    r = client.post("/v1/conversations", json={"purpose": "research"})
    assert r.status_code == 200
    conv_id = r.json()["id"]
    assert conv_id.startswith("conv-")

    # List
    r_list = client.get("/v1/conversations")
    assert r_list.status_code == 200
    ids = [c["id"] for c in r_list.json()["data"]]
    assert conv_id in ids

    # Delete
    r_del = client.delete(f"/v1/conversations/{conv_id}")
    assert r_del.status_code == 200
    assert r_del.json()["deleted"] is True


def test_chat_completions_non_streaming():
    fake_engine = FakeCompletionEngine(response_text="Test completed response.")
    app = create_app(engine=fake_engine)
    client = TestClient(app)

    payload = {
        "model": "gemini-3.8-flash",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": False,
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "Test completed response."
    assert data["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_streaming():
    fake_engine = FakeCompletionEngine(response_text="Streamed chunk output.")
    app = create_app(engine=fake_engine)
    client = TestClient(app)

    payload = {
        "model": "gemini-3.8-flash",
        "messages": [{"role": "user", "content": "Stream me!"}],
        "stream": True,
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]

    body = r.text
    assert ": ping" in body
    assert "data: " in body
    assert "[DONE]" in body
    assert "Streamed chunk output." in body


def test_unknown_model_returns_404():
    app = create_app(engine=FakeCompletionEngine())
    client = TestClient(app)

    payload = {
        "model": "nonexistent-model-xyz",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_unsupported_parameter_n_greater_than_1():
    app = create_app(engine=FakeCompletionEngine())
    client = TestClient(app)

    payload = {
        "model": "gemini-3.8-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "n": 2,
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_parameter"


def test_chat_completions_rate_limit_error_returns_429():
    err = RateLimitExceeded("Google account quota exhausted", retry_after_s=45)
    app = create_app(engine=FakeCompletionEngine(fails_with=err))
    client = TestClient(app)

    payload = {
        "model": "gemini-3.8-flash",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 429
    assert r.headers.get("retry-after") == "45"
    data = r.json()
    assert data["error"]["code"] == "rate_limit_exceeded"
    assert data["error"]["type"] == "upstream_error"
