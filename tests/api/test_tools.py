"""Tests for Phase 9: Tool Contracts & Multimodal Guards (Sections 13 & 14)."""
import pytest
from starlette.testclient import TestClient

from agy_bridge.api.app import create_app
from agy_bridge.api.tools import validate_and_parse_tool_calls
from agy_bridge.errors import ToolContractViolationError


def test_image_input_rejected_before_admission_returns_400():
    app = create_app()
    client = TestClient(app)

    payload = {
        "model": "gemini-3.8-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this picture?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},
                    },
                ],
            }
        ],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 400
    data = r.json()
    assert data["error"]["code"] == "unsupported_image_input"
    assert data["error"]["type"] == "invalid_request_error"


def test_valid_tool_calls_normalized():
    raw_calls = [
        {
            "name": "get_weather",
            "arguments": '{"location": "Binghamton, NY"}',
            "id": "call_123",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "description": "Get weather"},
        }
    ]
    parsed = validate_and_parse_tool_calls(raw_calls, declared_tools=tools)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "call_123"
    assert parsed[0]["function"]["name"] == "get_weather"
    assert parsed[0]["function"]["arguments"] == '{"location": "Binghamton, NY"}'


def test_undeclared_tool_call_rejected():
    raw_calls = [{"name": "execute_shell", "arguments": "{}"}]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    with pytest.raises(ToolContractViolationError, match="undeclared in tools schema"):
        validate_and_parse_tool_calls(raw_calls, declared_tools=tools)


def test_malformed_json_arguments_rejected():
    raw_calls = [{"name": "get_weather", "arguments": "{not_valid_json"}]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    with pytest.raises(ToolContractViolationError, match="not valid JSON"):
        validate_and_parse_tool_calls(raw_calls, declared_tools=tools)


def test_tool_choice_required_enforced():
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    with pytest.raises(ToolContractViolationError, match="Required tool call was not emitted"):
        validate_and_parse_tool_calls([], declared_tools=tools, tool_choice="required")


def test_tool_choice_none_enforced():
    raw_calls = [{"name": "get_weather", "arguments": "{}"}]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    with pytest.raises(ToolContractViolationError, match="tool_choice='none'"):
        validate_and_parse_tool_calls(raw_calls, declared_tools=tools, tool_choice="none")


def test_named_tool_choice_enforced():
    raw_calls = [{"name": "other_tool", "arguments": "{}"}]
    tools = [
        {"type": "function", "function": {"name": "other_tool"}},
        {"type": "function", "function": {"name": "target_tool"}},
    ]
    named_choice = {"type": "function", "function": {"name": "target_tool"}}

    with pytest.raises(ToolContractViolationError, match="Named tool 'target_tool'"):
        validate_and_parse_tool_calls(raw_calls, declared_tools=tools, tool_choice=named_choice)
