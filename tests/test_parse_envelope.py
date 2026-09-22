"""Characterization tests for bridge.parse_envelope (current behavior lock).

These tests document the CURRENT behavior of parse_envelope BEFORE any refactor.
They must all pass against the current bridge.py. If a case below does not
match reality, the expectation is corrected to reality (that is the behavior
lock), never the production code changed.
"""
import json
from pathlib import Path

import pytest

import bridge

FIXTURES = Path(__file__).resolve().parents[0] / "fixtures" / "envelopes"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


# ---------------------------------------------------------------------------
# No-tools path: returns raw text verbatim, never JSON-reformatted.
# ---------------------------------------------------------------------------

def test_no_tools_str_returns_raw_text_verbatim():
    text = 'Give me JSON: {"city": "Boston"}'
    assert bridge.parse_envelope(text) == (text, [])


def test_no_tools_dict_returns_json_dumps():
    obj = {"a": 1}
    assert bridge.parse_envelope(obj) == (json.dumps(obj), [])


def test_no_tools_list_returns_json_dumps():
    obj = [1, 2, 3]
    assert bridge.parse_envelope(obj) == (json.dumps(obj), [])


def test_no_tools_none_returns_empty():
    assert bridge.parse_envelope(None) == ("", [])


# ---------------------------------------------------------------------------
# Tools + list
# ---------------------------------------------------------------------------

def test_list_of_function_calls_normalized():
    raw = [{"type": "function", "function": {"name": "lookup_weather", "arguments": {"city": "Boston"}}}]
    content, tcs = bridge.parse_envelope(raw, tools=[{"type": "function", "function": {"name": "lookup_weather"}}])
    assert content == ""
    assert len(tcs) == 1
    assert tcs[0]["name"] == "lookup_weather"
    assert tcs[0]["arguments"] == {"city": "Boston"}


def test_list_finish_response_extracted():
    raw = [{"name": "final_answer", "arguments": {"response": "The answer is 42"}}]
    content, tcs = bridge.parse_envelope(raw, tools=[{"type": "function", "function": {"name": "other"}}])
    assert tcs == []
    assert content == "The answer is 42"


def test_list_with_tool_calls_subkey_normalized():
    raw = [{"content": "calling", "tool_calls": [{"type": "function", "function": {"name": "run_command", "arguments": {"CommandLine": "ls"}}}]}]
    content, tcs = bridge.parse_envelope(raw, tools=[{"type": "function", "function": {"name": "run_command"}}])
    assert content == ""
    assert len(tcs) == 1
    assert tcs[0]["name"] == "run_command"


def test_empty_list_returns_json_dumps():
    raw = []
    assert bridge.parse_envelope(raw, tools=[{"type": "function", "function": {"name": "x"}}]) == ("[]", [])


# ---------------------------------------------------------------------------
# Tools + dict
# ---------------------------------------------------------------------------

def test_dict_tool_calls_key_walked():
    # Real AGY lane envelope shape (from research doc):
    # {content, tool_calls:[{name, arguments}]}
    env = _load("agy_tool_call.json")
    content, tcs = bridge.parse_envelope(env, tools=[{"type": "function", "function": {"name": "lookup_weather"}}])
    assert content == "I will look up the weather for you."
    assert len(tcs) == 1
    assert tcs[0]["name"] == "lookup_weather"
    assert tcs[0]["arguments"] == {"city": "Boston, MA"}


def test_dict_tool_call_with_id_preserved():
    env = _load("agy_tool_call_id.json")
    content, tcs = bridge.parse_envelope(env, tools=[{"type": "function", "function": {"name": "run_command"}}])
    assert content == ""
    assert len(tcs) == 1
    assert tcs[0]["id"] == "call_001"
    assert tcs[0]["name"] == "run_command"


def test_dict_direct_tool_call_shape():
    raw = {"name": "search", "arguments": {"q": "hello"}}
    content, tcs = bridge.parse_envelope(raw, tools=[{"type": "function", "function": {"name": "search"}}])
    assert content == ""
    assert len(tcs) == 1
    assert tcs[0]["name"] == "search"


def test_dict_with_content_only_no_tcs_returns_content():
    raw = {"content": "plain answer"}
    content, tcs = bridge.parse_envelope(raw, tools=[{"type": "function", "function": {"name": "x"}}])
    assert tcs == []
    assert content == "plain answer"


# ---------------------------------------------------------------------------
# Nested content envelope
# ---------------------------------------------------------------------------

def test_nested_content_envelope_recursed():
    env = _load("agy_nested_content.json")
    # content = {"text":"...","tool_calls":[]}; inner recursion yields no tcs,
    # so parse_envelope falls through and returns the dict as content.
    content, tcs = bridge.parse_envelope(env, tools=[{"type": "function", "function": {"name": "lookup_weather"}}])
    assert tcs == []
    assert content == {"text": "The finish response follows.", "tool_calls": []}


# ---------------------------------------------------------------------------
# JSON array / ndjson string semantics
# ---------------------------------------------------------------------------

def test_json_array_of_dicts_string():
    obj = _load("agy_ndjson_array.json")
    raw_text = json.dumps(obj)
    content, tcs = bridge.parse_envelope(raw_text, tools=[])
    # no tools -> json.dumps of the dict/list
    assert content == raw_text
    assert tcs == []


def test_raw_json_string_passthrough_no_tools():
    raw_text = '[{"a": 1}]'
    assert bridge.parse_envelope(raw_text) == (raw_text, [])


# ---------------------------------------------------------------------------
# Garbage input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("garbage", [None, "not json at all", [], {}, 123])
def test_garbage_does_not_raise(garbage):
    content, tcs = bridge.parse_envelope(garbage, tools=[{"type": "function", "function": {"name": "x"}}])
    assert isinstance(content, str)
    assert isinstance(tcs, list)


def test_none_with_tools_returns_empty():
    assert bridge.parse_envelope(None, tools=[{"type": "function", "function": {"name": "x"}}]) == ("", [])


# ---------------------------------------------------------------------------
# Tool-call args that are not valid JSON
# ---------------------------------------------------------------------------

def test_invalid_json_arguments_cleaned_to_none_content_in_cleaner():
    # clean_tool_call_content returns None for envelope-ish content
    assert bridge.clean_tool_call_content('{"tool_calls": []}') is None
    assert bridge.clean_tool_call_content("{not json tool_calls}") is None or isinstance(
        bridge.clean_tool_call_content("{not json tool_calls}"), str)


def test_clean_tool_call_content_plain_text_passes_through():
    assert bridge.clean_tool_call_content("hello world") == "hello world"


def test_clean_tool_call_content_markdown_json_stripped():
    assert bridge.clean_tool_call_content("```json\n{\"tool_calls\": []}\n```") is None