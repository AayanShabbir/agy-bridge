"""Tests for Phase 9: Tools & Vision Policy (Section 13 & 14)."""
import pytest
from starlette.testclient import TestClient

from agy_bridge.api.app import create_app
from agy_bridge.api.tools import validate_and_parse_tool_calls, ToolContractViolationError


def test_vision_legacy_placeholder_reaches_engine_admission():
    app = create_app()
    client = TestClient(app)

    payload = {
        "model": "gemini-3.8-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/test.jpg"}},
                ],
            }
        ],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 503
    data = r.json()
    assert data["error"]["code"] == "upstream_unavailable"


def test_validate_and_parse_valid_tool_call():
    raw_calls = [
        {
            "name": "get_weather",
            "arguments": '{"location": "New York", "unit": "celsius"}',
        }
    ]
    declared_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    parsed = validate_and_parse_tool_calls(raw_calls, declared_tools)
    assert len(parsed) == 1
    assert parsed[0]["function"]["name"] == "get_weather"
    assert "location" in parsed[0]["function"]["arguments"]


def test_reject_undeclared_tool_call():
    raw_calls = [{"name": "unknown_tool", "arguments": "{}"}]
    declared_tools = [{"type": "function", "function": {"name": "known_tool"}}]

    with pytest.raises(ToolContractViolationError) as exc_info:
        validate_and_parse_tool_calls(raw_calls, declared_tools)
    assert "undeclared" in str(exc_info.value).lower()


def test_reject_malformed_json_tool_arguments():
    raw_calls = [{"name": "test_tool", "arguments": "{invalid-json"}]
    declared_tools = [{"type": "function", "function": {"name": "test_tool"}}]

    with pytest.raises(ToolContractViolationError) as exc_info:
        validate_and_parse_tool_calls(raw_calls, declared_tools)
    assert "json" in str(exc_info.value).lower()
